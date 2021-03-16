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
package org.deeplearning4j.nn.modelimport.keras.e2e;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.resources.Resources;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

/**
 * Test importing Keras models with multiple Lamdba layers.
 *
 * @author Max Pumperla
 */
@DisplayName("Keras Lambda Test")
class KerasLambdaTest extends BaseDL4JTest {

    @TempDir
    public Path testDir;

    @DisplayName("Exponential Lambda")
    class ExponentialLambda extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sd, SDVariable x) {
            return x.mul(x);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @DisplayName("Times Three Lambda")
    class TimesThreeLambda extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sd, SDVariable x) {
            return x.mul(3);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @Test
    @DisplayName("Test Sequential Lambda Layer Import")
    void testSequentialLambdaLayerImport() throws Exception {
        KerasLayer.registerLambdaLayer("lambda_1", new ExponentialLambda());
        KerasLayer.registerLambdaLayer("lambda_2", new TimesThreeLambda());
        String modelPath = "modelimport/keras/examples/lambda/sequential_lambda.h5";
        try (InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.resolve("tempModel" + System.currentTimeMillis() + ".h5").toFile();
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            MultiLayerNetwork model = new KerasSequentialModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(false).buildSequential().getMultiLayerNetwork();
            System.out.println(model.summary());
            INDArray input = Nd4j.create(new int[] { 10, 100 });
            model.output(input);
        } finally {
            KerasLayer.clearLambdaLayers();
        }
    }

    @Test
    @DisplayName("Test Model Lambda Layer Import")
    void testModelLambdaLayerImport() throws Exception {
        KerasLayer.registerLambdaLayer("lambda_3", new ExponentialLambda());
        KerasLayer.registerLambdaLayer("lambda_4", new TimesThreeLambda());
        String modelPath = "modelimport/keras/examples/lambda/model_lambda.h5";
        try (InputStream is = Resources.asStream(modelPath)) {
            File modelFile = testDir.resolve("tempModel" + System.currentTimeMillis() + ".h5").toFile();
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            ComputationGraph model = new KerasModel().modelBuilder().modelHdf5Filename(modelFile.getAbsolutePath()).enforceTrainingConfig(false).buildModel().getComputationGraph();
            System.out.println(model.summary());
            INDArray input = Nd4j.create(new int[] { 10, 784 });
            model.output(input);
        } finally {
            // Clear all lambdas, so other tests aren't affected.
            KerasLayer.clearLambdaLayers();
        }
    }
}
