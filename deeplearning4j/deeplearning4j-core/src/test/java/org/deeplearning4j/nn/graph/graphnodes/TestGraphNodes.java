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

package org.deeplearning4j.nn.graph.graphnodes;

import lombok.val;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.*;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;
import java.util.Map;

import static org.junit.Assert.*;

public class TestGraphNodes {

    @Test
    public void testMergeNode() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex mergeNode = new MergeVertex(null, "", -1);

        INDArray first = Nd4j.linspace(0, 11, 12).reshape(3, 4);
        INDArray second = Nd4j.linspace(0, 17, 18).reshape(3, 6).addi(100);

        mergeNode.setInputs(first, second);
        INDArray out = mergeNode.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(new long[] {3, 10}, out.shape());

        assertEquals(first, out.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));
        assertEquals(second, out.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 10)));

        mergeNode.setEpsilon(out);
        INDArray[] backward = mergeNode.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond();
        assertEquals(first, backward[0]);
        assertEquals(second, backward[1]);
    }

    @Test
    public void testMergeNodeRNN() {

        Nd4j.getRandom().setSeed(12345);
        GraphVertex mergeNode = new MergeVertex(null, "", -1);

        INDArray first = Nd4j.linspace(0, 59, 60).reshape(3, 4, 5);
        INDArray second = Nd4j.linspace(0, 89, 90).reshape(3, 6, 5).addi(100);

        mergeNode.setInputs(first, second);
        INDArray out = mergeNode.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(new long[] {3, 10, 5}, out.shape());

        assertEquals(first, out.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4), NDArrayIndex.all()));
        assertEquals(second, out.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 10), NDArrayIndex.all()));

        mergeNode.setEpsilon(out);
        INDArray[] backward = mergeNode.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond();
        assertEquals(first, backward[0]);
        assertEquals(second, backward[1]);
    }

    @Test
    public void testCnnDepthMerge() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex mergeNode = new MergeVertex(null, "", -1);

        INDArray first = Nd4j.linspace(0, 3, 4).reshape(1, 1, 2, 2);
        INDArray second = Nd4j.linspace(0, 3, 4).reshape(1, 1, 2, 2).addi(10);

        mergeNode.setInputs(first, second);
        INDArray out = mergeNode.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(new long[] {1, 2, 2, 2}, out.shape());

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(first.getDouble(0, 0, i, j), out.getDouble(0, 0, i, j), 1e-6);
                assertEquals(second.getDouble(0, 0, i, j), out.getDouble(0, 1, i, j), 1e-6);
            }
        }

        mergeNode.setEpsilon(out);
        INDArray[] backward = mergeNode.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond();
        assertEquals(first, backward[0]);
        assertEquals(second, backward[1]);


        //Slightly more complicated test:
        first = Nd4j.linspace(0, 17, 18).reshape(1, 2, 3, 3);
        second = Nd4j.linspace(0, 17, 18).reshape(1, 2, 3, 3).addi(100);

        mergeNode.setInputs(first, second);
        out = mergeNode.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertArrayEquals(new long[] {1, 4, 3, 3}, out.shape());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(first.getDouble(0, 0, i, j), out.getDouble(0, 0, i, j), 1e-6);
                assertEquals(first.getDouble(0, 1, i, j), out.getDouble(0, 1, i, j), 1e-6);

                assertEquals(second.getDouble(0, 0, i, j), out.getDouble(0, 2, i, j), 1e-6);
                assertEquals(second.getDouble(0, 1, i, j), out.getDouble(0, 3, i, j), 1e-6);
            }
        }

        mergeNode.setEpsilon(out);
        backward = mergeNode.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond();
        assertEquals(first, backward[0]);
        assertEquals(second, backward[1]);
    }

    @Test
    public void testSubsetNode() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex subset = new SubsetVertex(null, "", -1, 4, 7);

        INDArray in = Nd4j.rand(5, 10);
        subset.setInputs(in);
        INDArray out = subset.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(in.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 7, true)), out);

        subset.setEpsilon(out);
        INDArray backward = subset.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        assertEquals(Nd4j.zeros(5, 4), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true)));
        assertEquals(out, backward.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 7, true)));
        assertEquals(Nd4j.zeros(5, 2), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(8, 9, true)));

        //Test same for CNNs:
        in = Nd4j.rand(new int[] {5, 10, 3, 3});
        subset.setInputs(in);
        out = subset.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(in.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 7, true), NDArrayIndex.all(),
                        NDArrayIndex.all()), out);

        subset.setEpsilon(out);
        backward = subset.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        assertEquals(Nd4j.zeros(5, 4, 3, 3), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 3, true),
                        NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(out, backward.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 7, true), NDArrayIndex.all(),
                        NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 2, 3, 3), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(8, 9, true),
                        NDArrayIndex.all(), NDArrayIndex.all()));
    }


    @Test
    public void testLastTimeStepVertex() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                        .addVertex("lastTS", new LastTimeStepVertex("in"), "in")
                        .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).lossFunction(LossFunctions.LossFunction.MSE).build(), "lastTS").setOutputs("out")
                        .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        //First: test without input mask array
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(new int[] {3, 5, 6});
        INDArray expOut = in.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(5));

        GraphVertex gv = graph.getVertex("lastTS");
        gv.setInputs(in);
        //Forward pass:
        INDArray outFwd = gv.doForward(true, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);
        //Backward pass:
        gv.setEpsilon(expOut);
        Pair<Gradient, INDArray[]> pair = gv.doBackward(false, LayerWorkspaceMgr.noWorkspaces());
        INDArray eps = pair.getSecond()[0];
        assertArrayEquals(in.shape(), eps.shape());
        assertEquals(Nd4j.zeros(3, 5, 5),
                        eps.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4, true)));
        assertEquals(expOut, eps.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(5)));

        //Second: test with input mask array
        INDArray inMask = Nd4j.zeros(3, 6);
        inMask.putRow(0, Nd4j.create(new double[] {1, 1, 1, 0, 0, 0}));
        inMask.putRow(1, Nd4j.create(new double[] {1, 1, 1, 1, 0, 0}));
        inMask.putRow(2, Nd4j.create(new double[] {1, 1, 1, 1, 1, 0}));
        graph.setLayerMaskArrays(new INDArray[] {inMask}, null);

        expOut = Nd4j.zeros(3, 5);
        expOut.putRow(0, in.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(2)));
        expOut.putRow(1, in.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(3)));
        expOut.putRow(2, in.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.point(4)));

        gv.setInputs(in);
        outFwd = gv.doForward(true, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }

    @Test
    public void testDuplicateToTimeSeriesVertex() {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder()
                        .addInputs("in2d", "in3d")
                        .addVertex("duplicateTS", new DuplicateToTimeSeriesVertex("in3d"), "in2d")
                        .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).lossFunction(LossFunctions.LossFunction.MSE).build(), "duplicateTS")
                        .addLayer("out3d", new RnnOutputLayer.Builder().nIn(1).nOut(1).lossFunction(LossFunctions.LossFunction.MSE).build(), "in3d")
                        .setOutputs("out", "out3d").build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray in2d = Nd4j.rand(3, 5);
        INDArray in3d = Nd4j.rand(new int[] {3, 2, 7});

        graph.setInputs(in2d, in3d);

        INDArray expOut = Nd4j.zeros(3, 5, 7);
        for (int i = 0; i < 7; i++) {
            expOut.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(i)}, in2d);
        }

        GraphVertex gv = graph.getVertex("duplicateTS");
        gv.setInputs(in2d);
        INDArray outFwd = gv.doForward(true, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(expOut, outFwd);

        INDArray expOutBackward = expOut.sum(2);
        gv.setEpsilon(expOut);
        INDArray outBwd = gv.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        assertEquals(expOutBackward, outBwd);

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }

    @Test
    public void testStackNode() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex unstack = new StackVertex(null, "", -1);

        INDArray in1 = Nd4j.rand(5, 2);
        INDArray in2 = Nd4j.rand(5, 2);
        INDArray in3 = Nd4j.rand(5, 2);
        unstack.setInputs(in1, in2, in3);
        INDArray out = unstack.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(in1, out.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));
        assertEquals(in2, out.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));
        assertEquals(in3, out.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all()));

        unstack.setEpsilon(out);
        Pair<Gradient, INDArray[]> b = unstack.doBackward(false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(in1, b.getSecond()[0]);
        assertEquals(in2, b.getSecond()[1]);
        assertEquals(in3, b.getSecond()[2]);
    }

    @Test
    public void testStackVertexEmbedding() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex unstack = new StackVertex(null, "", -1);

        INDArray in1 = Nd4j.zeros(5, 1);
        INDArray in2 = Nd4j.zeros(5, 1);
        for (int i = 0; i < 5; i++) {
            in1.putScalar(i, 0, i);
            in2.putScalar(i, 0, i);
        }

        INDArray l = Nd4j.rand(5, 5);
        MultiDataSet ds = new org.nd4j.linalg.dataset.MultiDataSet(new INDArray[] {in1, in2}, new INDArray[] {l, l},
                        null, null);


        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in1", "in2")
                        .addVertex("stack", new org.deeplearning4j.nn.conf.graph.StackVertex(), "in1", "in2")
                        .addLayer("1", new EmbeddingLayer.Builder().nIn(5).nOut(5).build(), "stack")
                        .addVertex("unstack1", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "1")
                        .addVertex("unstack2", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 2), "1")
                        .addLayer("out1", new OutputLayer.Builder().activation(Activation.TANH)
                                        .lossFunction(LossFunctions.LossFunction.L2).nIn(5).nOut(5).build(), "unstack1")
                        .addLayer("out2", new OutputLayer.Builder().activation(Activation.TANH)
                                        .lossFunction(LossFunctions.LossFunction.L2).nIn(5).nOut(5).build(), "unstack2")
                        .setOutputs("out1", "out2").build();

        ComputationGraph g = new ComputationGraph(conf);
        g.init();

        g.feedForward(new INDArray[] {in1, in2}, false);

        g.fit(ds);

    }

    @Test
    public void testStackUnstackNodeVariableLength() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex stack = new StackVertex(null, "", -1);

        //Test stack with variable length + mask arrays
        INDArray in0 = Nd4j.rand(new int[] {5, 2, 5});
        INDArray in1 = Nd4j.rand(new int[] {5, 2, 6});
        INDArray in2 = Nd4j.rand(new int[] {5, 2, 7});

        INDArray mask0 = Nd4j.ones(5, 5);
        INDArray mask1 = Nd4j.ones(5, 6);
        INDArray mask2 = Nd4j.ones(5, 7);

        stack.setInputs(in0, in1, in2);
        Pair<INDArray, MaskState> p =
                        stack.feedForwardMaskArrays(new INDArray[] {mask0, mask1, mask2}, MaskState.Active, 5);
        assertArrayEquals(new long[] {15, 7}, p.getFirst().shape());
        assertEquals(MaskState.Active, p.getSecond());

        INDArray out = stack.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(in0, out.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all(), NDArrayIndex.interval(0, 5)));
        assertEquals(in1, out.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all(), NDArrayIndex.interval(0, 6)));
        assertEquals(in2, out.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all(), NDArrayIndex.interval(0, 7)));

        stack.setEpsilon(out);
        Pair<Gradient, INDArray[]> b = stack.doBackward(false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(in0, b.getSecond()[0]);
        assertEquals(in1, b.getSecond()[1]);
        assertEquals(in2, b.getSecond()[2]);

        //Test unstack with variable length + mask arrays
        //Note that we don't actually need changes here - unstack has a single input, and the unstacked mask
        //might be a bit longer than we really need, but it'll still be correct
        GraphVertex unstack0 = new UnstackVertex(null, "u0", 0, 0, 3);
        GraphVertex unstack1 = new UnstackVertex(null, "u1", 0, 1, 3);
        GraphVertex unstack2 = new UnstackVertex(null, "u2", 0, 2, 3);

        unstack0.setInputs(out);
        unstack1.setInputs(out);
        unstack2.setInputs(out);
        INDArray f0 = unstack0.doForward(true, LayerWorkspaceMgr.noWorkspaces());
        INDArray f1 = unstack1.doForward(true, LayerWorkspaceMgr.noWorkspaces());
        INDArray f2 = unstack2.doForward(true, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(in0, f0.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 5)));
        assertEquals(in1, f1.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 6)));
        assertEquals(in2, f2.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 7)));

        Pair<INDArray, MaskState> p0 =
                        unstack0.feedForwardMaskArrays(new INDArray[] {p.getFirst()}, MaskState.Active, 5);
        Pair<INDArray, MaskState> p1 =
                        unstack1.feedForwardMaskArrays(new INDArray[] {p.getFirst()}, MaskState.Active, 5);
        Pair<INDArray, MaskState> p2 =
                        unstack2.feedForwardMaskArrays(new INDArray[] {p.getFirst()}, MaskState.Active, 5);

        assertEquals(mask0, p0.getFirst().get(NDArrayIndex.all(), NDArrayIndex.interval(0, 5)));
        assertEquals(mask1, p1.getFirst().get(NDArrayIndex.all(), NDArrayIndex.interval(0, 6)));
        assertEquals(mask2, p2.getFirst().get(NDArrayIndex.all(), NDArrayIndex.interval(0, 7)));
    }

    @Test
    public void testUnstackNode() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex unstack0 = new UnstackVertex(null, "", -1, 0, 3);
        GraphVertex unstack1 = new UnstackVertex(null, "", -1, 1, 3);
        GraphVertex unstack2 = new UnstackVertex(null, "", -1, 2, 3);

        INDArray in = Nd4j.rand(15, 2);
        unstack0.setInputs(in);
        unstack1.setInputs(in);
        unstack2.setInputs(in);
        INDArray out0 = unstack0.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        INDArray out1 = unstack1.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        INDArray out2 = unstack2.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(in.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()), out0);
        assertEquals(in.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()), out1);
        assertEquals(in.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all()), out2);

        unstack0.setEpsilon(out0);
        unstack1.setEpsilon(out1);
        unstack2.setEpsilon(out2);
        INDArray backward0 = unstack0.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        INDArray backward1 = unstack1.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        INDArray backward2 = unstack2.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        assertEquals(out0, backward0.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 2), backward0.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 2), backward0.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5, 2), backward1.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));
        assertEquals(out1, backward1.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 2), backward1.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5, 2), backward2.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 2), backward2.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all()));
        assertEquals(out2, backward2.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all()));



        //Test same for CNNs:
        in = Nd4j.rand(new int[] {15, 10, 3, 3});
        unstack0.setInputs(in);
        unstack1.setInputs(in);
        unstack2.setInputs(in);
        out0 = unstack0.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        out1 = unstack1.doForward(false, LayerWorkspaceMgr.noWorkspaces());
        out2 = unstack2.doForward(false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(in.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()),
                        out0);
        assertEquals(in.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()),
                        out1);
        assertEquals(in.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()),
                        out2);

        unstack0.setEpsilon(out0);
        unstack1.setEpsilon(out1);
        unstack2.setEpsilon(out2);
        backward0 = unstack0.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        backward1 = unstack1.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        backward2 = unstack2.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond()[0];
        assertEquals(out0, backward0.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 10, 3, 3), backward0.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 10, 3, 3), backward0.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5, 10, 3, 3), backward1.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(out1, backward1.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 10, 3, 3), backward1.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5, 10, 3, 3), backward2.get(NDArrayIndex.interval(0, 5), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5, 10, 3, 3), backward2.get(NDArrayIndex.interval(5, 10), NDArrayIndex.all(),
                        NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(out2, backward2.get(NDArrayIndex.interval(10, 15), NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.all()));
    }

    @Test
    public void testL2Node() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex l2 = new L2Vertex(null, "", -1, 1e-8);

        INDArray in1 = Nd4j.rand(5, 2);
        INDArray in2 = Nd4j.rand(5, 2);

        l2.setInputs(in1, in2);
        INDArray out = l2.doForward(false, LayerWorkspaceMgr.noWorkspaces());

        INDArray expOut = Nd4j.create(5, 1);
        for (int i = 0; i < 5; i++) {
            double d2 = 0.0;
            for (int j = 0; j < in1.size(1); j++) {
                double temp = (in1.getDouble(i, j) - in2.getDouble(i, j));
                d2 += temp * temp;
            }
            d2 = Math.sqrt(d2);
            expOut.putScalar(i, 0, d2);
        }

        assertEquals(expOut, out);



        INDArray epsilon = Nd4j.rand(5, 1); //dL/dlambda
        INDArray diff = in1.sub(in2);
        //Out == sqrt(s) = s^1/2. Therefore: s^(-1/2) = 1/out
        INDArray sNegHalf = out.rdiv(1.0);

        INDArray dLda = diff.mulColumnVector(epsilon.mul(sNegHalf));
        INDArray dLdb = diff.mulColumnVector(epsilon.mul(sNegHalf)).neg();



        l2.setEpsilon(epsilon);
        Pair<Gradient, INDArray[]> p = l2.doBackward(false, LayerWorkspaceMgr.noWorkspaces());
        assertEquals(dLda, p.getSecond()[0]);
        assertEquals(dLdb, p.getSecond()[1]);
    }

    @Test
    public void testReshapeNode() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex reshapeVertex = new ReshapeVertex(null, "", -1, 'c', new int[] {-1, 736}, null);

        val inputShape = new long[] {1, 1, 1, 736};
        INDArray input = Nd4j.create(inputShape);

        reshapeVertex.setInputs(input);
        INDArray out = reshapeVertex.doForward(false, LayerWorkspaceMgr.noWorkspaces());

        assertArrayEquals(new long[] {1, 736}, out.shape());

        reshapeVertex.setEpsilon(out);
        INDArray[] backward = reshapeVertex.doBackward(false, LayerWorkspaceMgr.noWorkspaces()).getSecond();
        assertTrue(Arrays.equals(backward[0].shape(), inputShape));
    }

    @Test
    public void testJSON() {
        //The config here is non-sense, but that doesn't matter for config -> json -> config test
        ComputationGraphConfiguration conf =
                        new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                                        .addVertex("v1", new ElementWiseVertex(ElementWiseVertex.Op.Add), "in")
                                        .addVertex("v2", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "in", "in")
                                        .addVertex("v3", new PreprocessorVertex(
                                                        new CnnToFeedForwardPreProcessor(1, 2, 1)), "in")
                                        .addVertex("v4", new org.deeplearning4j.nn.conf.graph.SubsetVertex(0, 1), "in")
                                        .addVertex("v5", new DuplicateToTimeSeriesVertex("in"), "in")
                                        .addVertex("v6", new LastTimeStepVertex("in"), "in")
                                        .addVertex("v7", new org.deeplearning4j.nn.conf.graph.StackVertex(), "in")
                                        .addVertex("v8", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0, 1), "in")
                                        .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).lossFunction(LossFunctions.LossFunction.MSE).build(), "in")
                                        .setOutputs("out", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8").build();

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf, conf2);
    }


    @Test
    public void testLastTimeStepWithTransfer(){
        int lstmLayerSize = 16;
        int numLabelClasses = 10;
        int numInputs = 5;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .updater(new AdaDelta())
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("rr")
                .setInputTypes(InputType.recurrent(30))
                .addLayer("1", new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInputs).nOut(lstmLayerSize).dropOut(0.9).build(), "rr")
                .addLayer("2", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(numLabelClasses).build(), "1")

                .setOutputs("2")
                .build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        ComputationGraph updatedModel = new TransferLearning.GraphBuilder(net)
                .addVertex("laststepoutput", new LastTimeStepVertex("rr"), "2")
                .setOutputs("laststepoutput")
                .build();


        INDArray input = Nd4j.rand(new int[]{10, numInputs, 16});

        INDArray[] out = updatedModel.output(input);

        assertNotNull(out);
        assertEquals(1, out.length);
        assertNotNull(out[0]);

        assertArrayEquals(new long[]{10, numLabelClasses}, out[0].shape());

        Map<String,INDArray> acts = updatedModel.feedForward(input, false);

        assertEquals(4, acts.size());   //2 layers + input + vertex output
        assertNotNull(acts.get("laststepoutput"));
        assertArrayEquals(new long[]{10, numLabelClasses}, acts.get("laststepoutput").shape());

        String toString = out[0].toString();
    }
}
