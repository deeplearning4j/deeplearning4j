/*
 *  ******************************************************************************
 *  *
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
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.learning.config.AdaGrad;

import static org.deeplearning4j.nn.api.OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
import static org.deeplearning4j.nn.weights.WeightInit.XAVIER_UNIFORM;
import static org.junit.Assert.*;
import static org.nd4j.linalg.activations.Activation.IDENTITY;
import static org.nd4j.linalg.activations.Activation.TANH;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MSE;


@RunWith(Parameterized.class)
public class TestLastTimeStepLayer extends BaseDL4JTest {
    private RNNFormat rnnDataFormat;

    public TestLastTimeStepLayer(RNNFormat rnnDataFormat){
        this.rnnDataFormat = rnnDataFormat;
    }
    @Parameterized.Parameters(name="{0}")
    public static Object[] params(){
        return RNNFormat.values();
    }

    @Test
    public void testLastTimeStepVertex() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                .addLayer("lastTS", new LastTimeStep(new SimpleRnn.Builder()
                        .nIn(5).nOut(6).dataFormat(rnnDataFormat).build()), "in")
                .setOutputs("lastTS")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        //First: test without input mask array
        Nd4j.getRandom().setSeed(12345);
        Layer l = graph.getLayer("lastTS");
        INDArray in;
        if (rnnDataFormat == RNNFormat.NCW){
            in = Nd4j.rand(3, 5, 6);
        }
        else{
            in = Nd4j.rand(3, 6, 5);
        }
        INDArray outUnderlying = ((LastTimeStepLayer)l).getUnderlying().activate(in, false, LayerWorkspaceMgr.noWorkspaces());
        INDArray expOut;
        if (rnnDataFormat == RNNFormat.NCW){
            expOut = outUnderlying.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(5));
        }
        else{
            expOut = outUnderlying.get(NDArrayIndex.all(), NDArrayIndex.point(5), NDArrayIndex.all());
        }



        //Forward pass:
        INDArray outFwd = l.activate(in, false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);

        //Second: test with input mask array
        INDArray inMask = Nd4j.zeros(3, 6);
        inMask.putRow(0, Nd4j.create(new double[]{1, 1, 1, 0, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[]{1, 1, 1, 1, 0, 0}));
        inMask.putRow(2, Nd4j.create(new double[]{1, 1, 1, 1, 1, 0}));
        graph.setLayerMaskArrays(new INDArray[]{inMask}, null);

        expOut = Nd4j.zeros(3, 6);
        if (rnnDataFormat == RNNFormat.NCW){
            expOut.putRow(0, outUnderlying.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(2)));
            expOut.putRow(1, outUnderlying.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(3)));
            expOut.putRow(2, outUnderlying.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.point(4)));
        }
        else{
            expOut.putRow(0, outUnderlying.get(NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all()));
            expOut.putRow(1, outUnderlying.get(NDArrayIndex.point(1), NDArrayIndex.point(3), NDArrayIndex.all()));
            expOut.putRow(2, outUnderlying.get(NDArrayIndex.point(2), NDArrayIndex.point(4), NDArrayIndex.all()));
        }


        outFwd = l.activate(in, false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);

        TestUtils.testModelSerialization(graph);
    }

    @Test
    public void testMaskingAndAllMasked(){
        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(XAVIER_UNIFORM)
                .activation(TANH)
                .updater(new AdaGrad(0.01))
                .l2(0.0001)
                .seed(1234)
                .graphBuilder()
                .addInputs("in")
                .setInputTypes(InputType.recurrent(1, rnnDataFormat))
                .addLayer("RNN", new LastTimeStep(new LSTM.Builder()
                        .nOut(10).dataFormat(rnnDataFormat)
                        .build()), "in")
                .addLayer("dense", new DenseLayer.Builder()
                        .nOut(10)
                        .build(), "RNN")
                .addLayer("out", new OutputLayer.Builder()
                        .activation(IDENTITY)
                        .lossFunction(MSE)
                        .nOut(10)
                        .build(), "dense")
                .setOutputs("out");

        ComputationGraphConfiguration conf = builder.build();
        ComputationGraph cg = new ComputationGraph(conf);
        cg.init();

        INDArray f = Nd4j.rand(new long[]{1,1,24});
        INDArray fm1 = Nd4j.ones(1,24);
        INDArray fm2 = Nd4j.zeros(1,24);
        INDArray fm3 = Nd4j.zeros(1,24);
        fm3.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,5)).assign(1);
        if (rnnDataFormat == RNNFormat.NWC){
            f = f.permute(0, 2, 1);
        }
        INDArray[] out1 = cg.output(false, new INDArray[]{f}, new INDArray[]{fm1});
        try {
            cg.output(false, new INDArray[]{f}, new INDArray[]{fm2});
            fail("Expected exception");
        } catch (Exception e){
            assertTrue(e.getMessage().contains("mask is all 0s"));
        }

        INDArray[] out3 = cg.output(false, new INDArray[]{f}, new INDArray[]{fm3});

        System.out.println(out1[0]);
        System.out.println(out3[0]);

        assertNotEquals(out1[0], out3[0]);
    }
}
