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
package org.deeplearning4j.nn.modelimport.keras.configurations;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.InputStream;
import java.util.UUID;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Deep CTR Lambda Test")
class DeepCTRLambdaTest {

    @DisplayName("Tensors Sum")
    class TensorsSum extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            return layerInput.sum("tensors_sum-" + UUID.randomUUID().toString(), false, 1);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @DisplayName("Tensors Square")
    class TensorsSquare extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            return layerInput.mul("tensor_square-" + UUID.randomUUID().toString(), layerInput);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @DisplayName("Lambda 1")
    class Lambda1 extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            return layerInput.mul("lambda1-" + UUID.randomUUID().toString(), 0.5);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @DisplayName("Tensor Mean")
    class TensorMean extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            if (this.layerName.equals("concat_embed_2d") || this.layerName.equals("cat_embed_2d_genure_mean"))
                return layerInput.mean("mean_pooling-" + UUID.randomUUID().toString(), true, 1);
            else
                return layerInput.mean("mean_pooling-" + UUID.randomUUID().toString(), false, 1);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @Test
    @DisplayName("Test Deep Ctr")
    void testDeepCtr() throws Exception {
        KerasLayer.registerLambdaLayer("sum_of_tensors", new TensorsSum());
        KerasLayer.registerLambdaLayer("square_of_tensors", new TensorsSquare());
        KerasLayer.registerLambdaLayer("lambda_1", new Lambda1());
        KerasLayer.registerLambdaLayer("cat_embed_2d_genure_mean", new TensorMean());
        KerasLayer.registerLambdaLayer("embed_1d_mean", new TensorMean());
        ClassPathResource classPathResource = new ClassPathResource("modelimport/keras/examples/deepfm/deepfm.h5");
        try (InputStream inputStream = classPathResource.getInputStream();
            INDArray input0 = Nd4j.createNpyFromInputStream(new ClassPathResource("modelimport/keras/examples/deepfm/deepfm_x_0.npy").getInputStream());
            INDArray input1 = Nd4j.createNpyFromInputStream(new ClassPathResource("modelimport/keras/examples/deepfm/deepfm_x_1.npy").getInputStream());
            INDArray input2 = Nd4j.createNpyFromInputStream(new ClassPathResource("modelimport/keras/examples/deepfm/deepfm_x_2.npy").getInputStream());
            INDArray input3 = Nd4j.createNpyFromInputStream(new ClassPathResource("modelimport/keras/examples/deepfm/deepfm_x_3.npy").getInputStream())) {
            INDArray input0Reshaped = input0.reshape(input0.length(), 1);
            ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(inputStream);
            computationGraph.output(input0Reshaped, input1, input2, input3);
        }
    }
}
