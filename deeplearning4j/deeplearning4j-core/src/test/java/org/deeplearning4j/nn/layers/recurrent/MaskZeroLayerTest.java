/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class MaskZeroLayerTest extends BaseDL4JTest {
    private RNNFormat rnnDataFormat;

    public MaskZeroLayerTest(RNNFormat rnnDataFormat){
        this.rnnDataFormat = rnnDataFormat;
    }
    @Parameterized.Parameters
    public static Object[] params(){
        return RNNFormat.values();
    }
    @Test
    public void activate() {

        //GIVEN two examples where some of the timesteps are zero.
        INDArray ex1 = Nd4j.create(new double[][]{
                new double[]{0, 3, 5},
                new double[]{0, 0, 2}
        });
        INDArray ex2 = Nd4j.create(new double[][]{
                new double[]{0, 0, 2},
                new double[]{0, 0, 2}
        });

        // A LSTM which adds one for every non-zero timestep
        org.deeplearning4j.nn.conf.layers.LSTM underlying = new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                .activation(Activation.IDENTITY)
                .gateActivationFunction(Activation.IDENTITY)
                .nIn(2)
                .nOut(1).dataFormat(rnnDataFormat)
                .build();
        NeuralNetConfiguration conf = new NeuralNetConfiguration();
        conf.setLayer(underlying);
        INDArray params = Nd4j.zeros(new int[]{1, 16});

        //Set the biases to 1.
        for (int i = 12; i < 16; i++) {
            params.putScalar(i, 1.0);
        }
        Layer lstm = underlying.instantiate(conf, Collections.<TrainingListener>emptyList(), 0, params, false, params.dataType());
        double maskingValue = 0.0;

        MaskZeroLayer l = new MaskZeroLayer(lstm, maskingValue);
        INDArray input = Nd4j.create(Arrays.asList(ex1, ex2), new int[]{2, 2, 3});
        if (rnnDataFormat == RNNFormat.NWC){
            input = input.permute(0, 2, 1);
        }
        //WHEN
        INDArray out = l.activate(input, true, LayerWorkspaceMgr.noWorkspaces());
        if (rnnDataFormat == RNNFormat.NWC){
            out = out.permute(0, 2,1);
        }
        //THEN output should only be incremented for the non-zero timesteps
        INDArray firstExampleOutput = out.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray secondExampleOutput = out.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.all());

        assertEquals(0.0, firstExampleOutput.getDouble(0),  1e-6);
        assertEquals(1.0, firstExampleOutput.getDouble(1), 1e-6);
        assertEquals(2.0, firstExampleOutput.getDouble(2), 1e-6);

        assertEquals(0.0, secondExampleOutput.getDouble(0), 1e-6);
        assertEquals(0.0, secondExampleOutput.getDouble(1),  1e-6);
        assertEquals(1.0, secondExampleOutput.getDouble(2), 1e-6);

    }

    @Test
    public void testSerialization(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new org.deeplearning4j.nn.conf.layers.util.MaskZeroLayer.Builder()
                        .setMaskValue(0.0).setUnderlying(new LSTM.Builder().nIn(4).nOut(5).dataFormat(rnnDataFormat).build()).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        TestUtils.testModelSerialization(net);
    }
}
