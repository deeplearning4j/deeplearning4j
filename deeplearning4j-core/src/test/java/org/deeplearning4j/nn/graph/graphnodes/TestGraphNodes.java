package org.deeplearning4j.nn.graph.graphnodes;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.ElementWiseVertex;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class TestGraphNodes {

    @Test
    public void testMergeNode() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex mergeNode = new MergeVertex(null,"",-1);

        INDArray first = Nd4j.linspace(0, 11, 12).reshape(3, 4);
        INDArray second = Nd4j.linspace(0, 17, 18).reshape(3, 6).addi(100);

        mergeNode.setInputs(first, second);
        INDArray out = mergeNode.doForward(false);
        assertArrayEquals(new int[]{3, 10}, out.shape());

        assertEquals(first, out.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4)));
        assertEquals(second, out.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 10)));

        mergeNode.setEpsilon(out);
        INDArray[] backward = mergeNode.doBackward(false).getSecond();
        assertEquals(first,backward[0]);
        assertEquals(second, backward[1]);
    }

    @Test
    public void testMergeNodeRNN() {

        Nd4j.getRandom().setSeed(12345);
        GraphVertex mergeNode = new MergeVertex(null,"",-1);

        INDArray first = Nd4j.linspace(0, 59, 60).reshape(3, 4, 5);
        INDArray second = Nd4j.linspace(0, 89, 90).reshape(3, 6, 5).addi(100);

        mergeNode.setInputs(first, second);
        INDArray out = mergeNode.doForward(false);
        assertArrayEquals(new int[]{3, 10, 5}, out.shape());

        assertEquals(first, out.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 4), NDArrayIndex.all()));
        assertEquals(second, out.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 10), NDArrayIndex.all()));

        mergeNode.setEpsilon(out);
        INDArray[] backward = mergeNode.doBackward(false).getSecond();
        assertEquals(first,backward[0]);
        assertEquals(second, backward[1]);
    }

    @Test
    public void testCnnDepthMerge() {
        Nd4j.getRandom().setSeed(12345);
        GraphVertex mergeNode = new MergeVertex(null,"",-1);

        INDArray first = Nd4j.linspace(0, 3, 4).reshape(1, 1, 2, 2);
        INDArray second = Nd4j.linspace(0, 3, 4).reshape(1, 1, 2, 2).addi(10);

        mergeNode.setInputs(first, second);
        INDArray out = mergeNode.doForward(false);
        assertArrayEquals(new int[]{1, 2, 2, 2}, out.shape());

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(first.getDouble(0, 0, i, j), out.getDouble(0, 0, i, j), 1e-6);
                assertEquals(second.getDouble(0, 0, i, j), out.getDouble(0, 1, i, j), 1e-6);
            }
        }

        mergeNode.setEpsilon(out);
        INDArray[] backward = mergeNode.doBackward(false).getSecond();
        assertEquals(first, backward[0]);
        assertEquals(second, backward[1]);


        //Slightly more complicated test:
        first = Nd4j.linspace(0, 17, 18).reshape(1, 2, 3, 3);
        second = Nd4j.linspace(0, 17, 18).reshape(1, 2, 3, 3).addi(100);

        mergeNode.setInputs(first,second);
        out = mergeNode.doForward(false);
        assertArrayEquals(new int[]{1, 4, 3, 3}, out.shape());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(first.getDouble(0, 0, i, j), out.getDouble(0, 0, i, j), 1e-6);
                assertEquals(first.getDouble(0, 1, i, j), out.getDouble(0, 1, i, j), 1e-6);

                assertEquals(second.getDouble(0, 0, i, j), out.getDouble(0, 2, i, j), 1e-6);
                assertEquals(second.getDouble(0, 1, i, j), out.getDouble(0, 3, i, j), 1e-6);
            }
        }

        mergeNode.setEpsilon(out);
        backward = mergeNode.doBackward(false).getSecond();
        assertEquals(first, backward[0]);
        assertEquals(second, backward[1]);
    }

    @Test
    public void testSubsetNode(){
        Nd4j.getRandom().setSeed(12345);
        GraphVertex subset = new SubsetVertex(null,"",-1,4,7);

        INDArray in = Nd4j.rand(5, 10);
        subset.setInputs(in);
        INDArray out = subset.doForward(false);
        assertEquals(in.get(NDArrayIndex.all(), NDArrayIndex.interval(4, 7, true)), out);

        subset.setEpsilon(out);
        INDArray backward = subset.doBackward(false).getSecond()[0];
        assertEquals(Nd4j.zeros(5,4),backward.get(NDArrayIndex.all(),NDArrayIndex.interval(0,3,true)));
        assertEquals(out, backward.get(NDArrayIndex.all(), NDArrayIndex.interval(4,7,true)));
        assertEquals(Nd4j.zeros(5,2), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(8,9,true)));

        //Test same for CNNs:
        in = Nd4j.rand(new int[]{5, 10, 3, 3});
        subset.setInputs(in);
        out = subset.doForward(false);
        assertEquals(in.get(NDArrayIndex.all(),NDArrayIndex.interval(4,7,true), NDArrayIndex.all(), NDArrayIndex.all()),out);

        subset.setEpsilon(out);
        backward = subset.doBackward(false).getSecond()[0];
        assertEquals(Nd4j.zeros(5,4,3,3),backward.get(NDArrayIndex.all(),NDArrayIndex.interval(0,3,true), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(out, backward.get(NDArrayIndex.all(), NDArrayIndex.interval(4,7,true), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,2,3,3), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(8,9,true), NDArrayIndex.all(), NDArrayIndex.all()));
    }


    @Test
    public void testLastTimeStepVertex(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addVertex("lastTS", new LastTimeStepVertex("in"), "in")
                .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).build(), "lastTS")
                .setOutputs("out")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        //First: test without input mask array
        Nd4j.getRandom().setSeed(12345);
        INDArray in = Nd4j.rand(new int[]{3, 5, 6});
        INDArray expOut = in.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(5));

        GraphVertex gv = graph.getVertex("lastTS");
        gv.setInputs(in);
            //Forward pass:
        INDArray outFwd = gv.doForward(true);
        assertEquals(expOut, outFwd);
            //Backward pass:
        gv.setEpsilon(expOut);
        Pair<Gradient,INDArray[]> pair = gv.doBackward(false);
        INDArray eps = pair.getSecond()[0];
        assertArrayEquals(in.shape(), eps.shape());
        assertEquals(Nd4j.zeros(3, 5, 5), eps.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 4, true)));
        assertEquals(expOut, eps.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(5)));

        //Second: test with input mask array
        INDArray inMask = Nd4j.zeros(3,6);
        inMask.putRow(0,Nd4j.create(new double[]{1,1,1,0,0,0}));
        inMask.putRow(1,Nd4j.create(new double[]{1,1,1,1,0,0}));
        inMask.putRow(2, Nd4j.create(new double[]{1, 1, 1, 1, 1, 0}));
        graph.setLayerMaskArrays(new INDArray[]{inMask}, null);

        expOut = Nd4j.zeros(3,5);
        expOut.putRow(0,in.get(NDArrayIndex.point(0),NDArrayIndex.all(),NDArrayIndex.point(2)));
        expOut.putRow(1,in.get(NDArrayIndex.point(1),NDArrayIndex.all(),NDArrayIndex.point(3)));
        expOut.putRow(2, in.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.point(4)));

        gv.setInputs(in);
        outFwd = gv.doForward(true);
        assertEquals(expOut, outFwd);

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf,conf2);
    }

    @Test
    public void testDuplicateToTimeSeriesVertex(){

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in2d","in3d")
                .addVertex("duplicateTS", new DuplicateToTimeSeriesVertex("in3d"), "in2d")
                .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).build(), "duplicateTS")
                .setOutputs("out")
                .build();

        ComputationGraph graph = new ComputationGraph(conf);
        graph.init();

        INDArray in2d = Nd4j.rand(3,5);
        INDArray in3d = Nd4j.rand(new int[]{3,2,7});

        graph.setInputs(in2d,in3d);

        INDArray expOut = Nd4j.zeros(3,5,7);
        for( int i=0; i<7; i++){
            expOut.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(i)},in2d);
        }

        GraphVertex gv = graph.getVertex("duplicateTS");
        gv.setInputs(in2d);
        INDArray outFwd = gv.doForward(true);
        assertEquals(expOut, outFwd);

        INDArray expOutBackward = expOut.sum(2);
        gv.setEpsilon(expOut);
        INDArray outBwd = gv.doBackward(false).getSecond()[0];
        assertEquals(expOutBackward, outBwd);

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf,conf2);
    }

    @Test
    public void testStackNode(){
        Nd4j.getRandom().setSeed(12345);
        GraphVertex unstack = new StackVertex(null,"",-1);

        INDArray in1 = Nd4j.rand(5,2);
        INDArray in2 = Nd4j.rand(5,2);
        INDArray in3 = Nd4j.rand(5,2);
        unstack.setInputs(in1, in2, in3);
        INDArray out = unstack.doForward(false);
        assertEquals(in1, out.get(NDArrayIndex.interval(0,5), NDArrayIndex.all()));
        assertEquals(in2, out.get(NDArrayIndex.interval(5,10), NDArrayIndex.all()));
        assertEquals(in3, out.get(NDArrayIndex.interval(10,15), NDArrayIndex.all()));

        unstack.setErrors(out);
        Pair<Gradient,INDArray[]> b = unstack.doBackward(false);

        assertEquals(in1, b.getSecond()[0]);
        assertEquals(in2, b.getSecond()[1]);
        assertEquals(in3, b.getSecond()[2]);
    }

    @Test
    public void testUnstackNode(){
        Nd4j.getRandom().setSeed(12345);
        GraphVertex unstack0 = new UnstackVertex(null,"",-1,0,3);
        GraphVertex unstack1 = new UnstackVertex(null,"",-1,1,3);
        GraphVertex unstack2 = new UnstackVertex(null,"",-1,2,3);

        INDArray in = Nd4j.rand(15,2);
        unstack0.setInputs(in);
        unstack1.setInputs(in);
        unstack2.setInputs(in);
        INDArray out0 = unstack0.doForward(false);
        INDArray out1 = unstack1.doForward(false);
        INDArray out2 = unstack2.doForward(false);
        assertEquals(in.get(NDArrayIndex.interval(0,5), NDArrayIndex.all()), out0);
        assertEquals(in.get(NDArrayIndex.interval(5,10), NDArrayIndex.all()), out1);
        assertEquals(in.get(NDArrayIndex.interval(10,15), NDArrayIndex.all()), out2);

        unstack0.setErrors(out0);
        unstack1.setErrors(out1);
        unstack2.setErrors(out2);
        INDArray backward0 = unstack0.doBackward(false).getSecond()[0];
        INDArray backward1 = unstack1.doBackward(false).getSecond()[0];
        INDArray backward2 = unstack2.doBackward(false).getSecond()[0];
        assertEquals(out0, backward0.get(NDArrayIndex.interval(0,5), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,2), backward0.get(NDArrayIndex.interval(5,10), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,2), backward0.get(NDArrayIndex.interval(10,15), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5,2), backward1.get(NDArrayIndex.interval(0,5), NDArrayIndex.all()));
        assertEquals(out1, backward1.get(NDArrayIndex.interval(5,10), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,2), backward1.get(NDArrayIndex.interval(10,15), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5,2), backward2.get(NDArrayIndex.interval(0,5), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,2), backward2.get(NDArrayIndex.interval(5,10), NDArrayIndex.all()));
        assertEquals(out2, backward2.get(NDArrayIndex.interval(10,15), NDArrayIndex.all()));




        //Test same for CNNs:
        in = Nd4j.rand(new int[]{15, 10, 3, 3});
        unstack0.setInputs(in);
        unstack1.setInputs(in);
        unstack2.setInputs(in);
        out0 = unstack0.doForward(false);
        out1 = unstack1.doForward(false);
        out2 = unstack2.doForward(false);

        assertEquals(in.get(NDArrayIndex.interval(0,5), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()), out0);
        assertEquals(in.get(NDArrayIndex.interval(5,10), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()), out1);
        assertEquals(in.get(NDArrayIndex.interval(10,15), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()), out2);

        unstack0.setErrors(out0);
        unstack1.setErrors(out1);
        unstack2.setErrors(out2);
        backward0 = unstack0.doBackward(false).getSecond()[0];
        backward1 = unstack1.doBackward(false).getSecond()[0];
        backward2 = unstack2.doBackward(false).getSecond()[0];
        assertEquals(out0, backward0.get(NDArrayIndex.interval(0,5), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,10,3,3), backward0.get(NDArrayIndex.interval(5,10), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,10,3,3), backward0.get(NDArrayIndex.interval(10,15), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5,10,3,3), backward1.get(NDArrayIndex.interval(0,5), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(out1, backward1.get(NDArrayIndex.interval(5,10), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,10,3,3), backward1.get(NDArrayIndex.interval(10,15), NDArrayIndex.all()));

        assertEquals(Nd4j.zeros(5,10,3,3), backward2.get(NDArrayIndex.interval(0,5), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(Nd4j.zeros(5,10,3,3), backward2.get(NDArrayIndex.interval(5,10), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
        assertEquals(out2, backward2.get(NDArrayIndex.interval(10,15), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()));
    }

    @Test
    public void testL2Node(){
        Nd4j.getRandom().setSeed(12345);
        GraphVertex l2 = new L2Vertex(null,"",-1, 1e-8);

        INDArray in1 = Nd4j.rand(5,2);
        INDArray in2 = Nd4j.rand(5,2);

        l2.setInputs(in1, in2);
        INDArray out = l2.doForward(false);

        INDArray expOut = Nd4j.create(5,1);
        for( int i=0; i<5; i++ ){
            double d2 = 0.0;
            for( int j=0; j<in1.size(1); j++ ){
                double temp = (in1.getDouble(i,j) - in2.getDouble(i,j));
                d2 += temp * temp;
            }
            d2 = Math.sqrt(d2);
            expOut.putScalar(i,0,d2);
        }

        assertEquals(expOut, out);



        INDArray epsilon = Nd4j.rand(5,1);      //dL/dlambda
        INDArray diff = in1.sub(in2);
        //Out == sqrt(s) = s^1/2. Therefore: s^(-1/2) = 1/out
        INDArray sNegHalf = out.rdiv(1.0);

        INDArray dLda = diff.mulColumnVector(epsilon.mul(sNegHalf));
        INDArray dLdb = diff.mulColumnVector(epsilon.mul(sNegHalf)).neg();



        l2.setErrors(epsilon);
        Pair<Gradient,INDArray[]> p = l2.doBackward(false);
        assertEquals(dLda, p.getSecond()[0]);
        assertEquals(dLdb, p.getSecond()[1]);
    }

    @Test
    public void testJSON(){
        //The config here is non-sense, but that doesn't matter for config -> json -> config test
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .graphBuilder()
            .addInputs("in")
            .addVertex("v1",new ElementWiseVertex(ElementWiseVertex.Op.Add),"in")
            .addVertex("v2", new org.deeplearning4j.nn.conf.graph.MergeVertex(), "in","in")
            .addVertex("v3", new PreprocessorVertex(new CnnToFeedForwardPreProcessor(1,2,1)), "in")
            .addVertex("v4", new org.deeplearning4j.nn.conf.graph.SubsetVertex(0,1),"in")
            .addVertex("v5", new DuplicateToTimeSeriesVertex("in"),"in")
            .addVertex("v6", new LastTimeStepVertex("in"), "in")
            .addVertex("v7", new org.deeplearning4j.nn.conf.graph.StackVertex(), "in")
            .addVertex("v8", new org.deeplearning4j.nn.conf.graph.UnstackVertex(0,1), "in")
            .addLayer("out", new OutputLayer.Builder().nIn(1).nOut(1).build(), "in")
            .setOutputs("out")
            .build();

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf,conf2);
    }
}
