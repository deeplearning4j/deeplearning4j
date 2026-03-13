/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.dl4jcore.nn.transferlearning;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@DisplayName("Transfer Learning Comp Graph Isolation Test")
class TransferLearningCompGraphIsolationTest extends BaseDL4JTest {

    @Test
    @DisplayName("Simple Fine Tune Forward Pass Executes")
    void simpleFineTuneForwardPassExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        INDArray output = models.modelNow.output(models.randomData.getFeatures())[0];
        assertNotNull(output);
        assertEquals(10, output.size(0));
        assertEquals(3, output.size(1));
    }

    @Test
    @DisplayName("Simple Fine Tune Expected Model Gradient Computation Executes")
    void simpleFineTuneExpectedModelGradientComputationExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        models.expectedModel.setInputs(models.randomData.getFeatures());
        models.expectedModel.setLabels(models.randomData.getLabels());
        models.expectedModel.computeGradientAndScore();

        Gradient gradient = models.expectedModel.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(models.expectedModel.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Simple Fine Tune Transfer Learning Gradient Views Match Expected Model")
    void simpleFineTuneTransferLearningGradientViewsMatchExpectedModel() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        models.expectedModel.initGradientsView();
        models.modelNow.initGradientsView();

        assertArrayEquals(models.expectedModel.getGradientsViewArray().shape(), models.modelNow.getGradientsViewArray().shape());
        assertArrayEquals(models.expectedModel.getGradientsViewArray().stride(), models.modelNow.getGradientsViewArray().stride());

        for (String layerName : new String[]{"layer0", "layer1"}) {
            assertArrayEquals(models.expectedModel.getLayer(layerName).getGradientsViewArray().shape(),
                    models.modelNow.getLayer(layerName).getGradientsViewArray().shape());
            assertArrayEquals(models.expectedModel.getLayer(layerName).getGradientsViewArray().stride(),
                    models.modelNow.getLayer(layerName).getGradientsViewArray().stride());
            assertArrayEquals(models.expectedModel.getLayer(layerName).params().shape(),
                    models.modelNow.getLayer(layerName).params().shape());
            assertArrayEquals(models.expectedModel.getLayer(layerName).params().stride(),
                    models.modelNow.getLayer(layerName).params().stride());
        }
    }

    @Test
    @DisplayName("Simple Fine Tune Transfer Learning Gradient Computation Executes")
    void simpleFineTuneTransferLearningGradientComputationExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();

        models.modelNow.setInputs(models.randomData.getFeatures());
        models.modelNow.setLabels(models.randomData.getLabels());
        models.modelNow.computeGradientAndScore();

        Gradient gradient = models.modelNow.gradient();
        assertNotNull(gradient);
        assertNotNull(gradient.gradient());
        assertEquals(models.modelNow.numParams(), gradient.gradient().length());
    }

    @Test
    @DisplayName("Simple Fine Tune Updater Initialization Executes")
    void simpleFineTuneUpdaterInitializationExecutes() {
        SimpleFineTuneModels models = simpleFineTuneModels();
        assertNotNull(models.modelNow.getUpdater());
    }

    private static SimpleFineTuneModels simpleFineTuneModels() {
        long rng = 12345L;
        DataSet randomData = new DataSet(Nd4j.rand(10, 4), Nd4j.rand(10, 3));

        ComputationGraphConfiguration expectedConf = new NeuralNetConfiguration.Builder().seed(rng)
                .updater(new RmsProp(0.2))
                .graphBuilder().addInputs("layer0In")
                .setInputTypes(InputType.feedForward(4))
                .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).build(), "layer0In")
                .addLayer("layer1", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(3).nOut(3).build(), "layer0")
                .setOutputs("layer1").build();

        ComputationGraph expectedModel = new ComputationGraph(expectedConf);
        expectedModel.init();

        ComputationGraph modelToFineTune = new ComputationGraph(expectedConf);
        modelToFineTune.init();
        modelToFineTune.setParams(expectedModel.params());

        ComputationGraph modelNow = new TransferLearning.GraphBuilder(modelToFineTune)
                .fineTuneConfiguration(new FineTuneConfiguration.Builder().seed(rng).updater(new RmsProp(0.2)).build())
                .build();

        return new SimpleFineTuneModels(randomData, expectedModel, modelNow);
    }

    private static final class SimpleFineTuneModels {
        private final DataSet randomData;
        private final ComputationGraph expectedModel;
        private final ComputationGraph modelNow;

        private SimpleFineTuneModels(DataSet randomData, ComputationGraph expectedModel, ComputationGraph modelNow) {
            this.randomData = randomData;
            this.expectedModel = expectedModel;
            this.modelNow = modelNow;
        }
    }
}
