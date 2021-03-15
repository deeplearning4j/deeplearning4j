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
package org.deeplearning4j.nn.layers.capsule;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Capsule Strength Layer Test")
class CapsuleStrengthLayerTest extends BaseDL4JTest {

    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Test
    @DisplayName("Test Output Type")
    void testOutputType() {
        CapsuleStrengthLayer layer = new CapsuleStrengthLayer.Builder().build();
        InputType in1 = InputType.recurrent(5, 8);
        assertEquals(InputType.feedForward(5), layer.getOutputType(0, in1));
    }

    @Test
    @DisplayName("Test Layer")
    void testLayer() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123).list().layer(new CapsuleStrengthLayer.Builder().build()).setInputType(InputType.recurrent(5, 8)).build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        INDArray emptyFeatures = Nd4j.zeros(64, 5, 10);
        long[] shape = model.output(emptyFeatures).shape();
        assertArrayEquals(new long[] { 64, 5 }, shape);
    }
}
