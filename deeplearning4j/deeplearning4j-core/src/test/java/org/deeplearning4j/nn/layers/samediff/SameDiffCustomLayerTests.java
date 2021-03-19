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

package org.deeplearning4j.nn.layers.samediff;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.conf.layers.samediff.SDVertexParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;

@Slf4j
public class SameDiffCustomLayerTests extends BaseDL4JTest {
    private DataType initialType;


    @BeforeEach
    public void before() {
        Nd4j.create(1);
        initialType = Nd4j.dataType();

        Nd4j.setDataType(DataType.DOUBLE);
        Nd4j.getRandom().setSeed(123);
    }

    @AfterEach
    public void after() {
        Nd4j.setDataType(initialType);

        NativeOpsHolder.getInstance().getDeviceNativeOps().enableDebugMode(false);
        NativeOpsHolder.getInstance().getDeviceNativeOps().enableVerboseMode(false);
    }

    @Test
    public void testInputValidationSameDiffLayer(){
        assertThrows(IllegalArgumentException.class,() -> {
            final MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().list()
                    .layer(new ValidatingSameDiffLayer())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nOut(2).build())
                    .setInputType(InputType.feedForward(2))
                    .build();

            final MultiLayerNetwork net = new MultiLayerNetwork(config);
            net.init();

            final INDArray goodInput = Nd4j.rand(1, 2);
            final INDArray badInput = Nd4j.rand(2, 2);

            net.fit(goodInput, goodInput);
            net.fit(badInput, badInput);


        });

    }

    @Test
    public void testInputValidationSameDiffVertex(){
       assertThrows(IllegalArgumentException.class,() -> {
           final ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder().graphBuilder()
                   .addVertex("a", new ValidatingSameDiffVertex(), "input")
                   .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nOut(2).build(), "a")
                   .addInputs("input")
                   .setInputTypes(InputType.feedForward(2))
                   .setOutputs("output")
                   .build();

           final ComputationGraph net = new ComputationGraph(config);
           net.init();

           final INDArray goodInput = Nd4j.rand(1, 2);
           final INDArray badInput = Nd4j.rand(2, 2);

           net.fit(new INDArray[]{goodInput}, new INDArray[]{goodInput});
           net.fit(new INDArray[]{badInput}, new INDArray[]{badInput});
       });

    }

    private class ValidatingSameDiffLayer extends org.deeplearning4j.nn.conf.layers.samediff.SameDiffLayer {
        @Override
        public void validateInput(INDArray input) {
            Preconditions.checkArgument(input.size(0) < 2, "Expected Message");
        }

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable, SDVariable mask) {
            return layerInput;
        }

        @Override
        public void defineParameters(SDLayerParams params) { }

        @Override
        public void initializeParameters(Map<String, INDArray> params) { }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) { return inputType; }
    }

    private class ValidatingSameDiffVertex extends SameDiffVertex {
        @Override
        public GraphVertex clone() {
            return new ValidatingSameDiffVertex();
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType... vertexInputs) throws InvalidInputTypeException {
            return vertexInputs[0];
        }

        @Override
        public void validateInput(INDArray[] input) {
            Preconditions.checkArgument(input[0].size(0) < 2, "Expected Message");
        }

        @Override
        public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable, Map<String, SDVariable> maskVars) {
            return layerInput.get("input");
        }

        @Override
        public void defineParametersAndInputs(SDVertexParams params) {
            params.defineInputs("input");
        }

        @Override
        public void initializeParameters(Map<String, INDArray> params) {}
    }
}
