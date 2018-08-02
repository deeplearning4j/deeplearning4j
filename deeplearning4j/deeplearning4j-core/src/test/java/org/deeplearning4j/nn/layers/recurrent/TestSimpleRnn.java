/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertEquals;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class TestSimpleRnn extends BaseDL4JTest {

    @Test
    public void testSimpleRnn(){
        Nd4j.getRandom().setSeed(12345);

        int m = 3;
        int nIn = 5;
        int layerSize = 6;
        int tsLength = 7;
        INDArray in = Nd4j.rand(new int[]{m, nIn, tsLength});
//        in.get(all(), all(), interval(1,tsLength)).assign(0);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new NoOp())
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .list()
                .layer(new SimpleRnn.Builder().nIn(nIn).nOut(layerSize).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        INDArray out = net.output(in);

        INDArray w = net.getParam("0_W");
        INDArray rw = net.getParam("0_RW");
        INDArray b = net.getParam("0_b");

        INDArray outLast = null;
        for( int i=0; i<tsLength; i++ ){
            INDArray inCurrent = in.get(all(), all(), point(i));

            INDArray outExpCurrent = inCurrent.mmul(w);
            if(outLast != null){
                outExpCurrent.addi(outLast.mmul(rw));
            }

            outExpCurrent.addiRowVector(b);

            Transforms.tanh(outExpCurrent, false);

            INDArray outActCurrent = out.get(all(), all(), point(i));
            assertEquals(String.valueOf(i), outExpCurrent, outActCurrent);

            outLast = outExpCurrent;
        }


        TestUtils.testModelSerialization(net);
    }

}
