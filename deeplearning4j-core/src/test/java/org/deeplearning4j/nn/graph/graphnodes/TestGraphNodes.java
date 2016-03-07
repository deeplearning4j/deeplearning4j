package org.deeplearning4j.nn.graph.graphnodes;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.impl.MergeVertex;
import org.deeplearning4j.nn.graph.vertex.impl.SubsetVertex;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

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

        mergeNode.setErrors(out);
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

        mergeNode.setErrors(out);
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

        mergeNode.setErrors(out);
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

        mergeNode.setErrors(out);
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

        subset.setErrors(out);
        INDArray backward = subset.doBackward(false).getSecond()[0];
        assertEquals(Nd4j.zeros(5,4),backward.get(NDArrayIndex.all(),NDArrayIndex.interval(0,3,true)));
        assertEquals(out, backward.get(NDArrayIndex.all(), NDArrayIndex.interval(4,7,true)));
        assertEquals(Nd4j.zeros(5,2), backward.get(NDArrayIndex.all(), NDArrayIndex.interval(8,9,true)));

        //Test same for CNNs:
        in = Nd4j.rand(new int[]{5, 10, 3, 3});
        subset.setInputs(in);
        out = subset.doForward(false);
        assertEquals(in.get(NDArrayIndex.all(),NDArrayIndex.interval(4,7,true), NDArrayIndex.all(), NDArrayIndex.all()),out);

        subset.setErrors(out);
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
        gv.setError(0,expOut);
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
        gv.setError(0, expOut);
        INDArray outBwd = gv.doBackward(false).getSecond()[0];
        assertEquals(expOutBackward, outBwd);

        String json = conf.toJson();
        ComputationGraphConfiguration conf2 = ComputationGraphConfiguration.fromJson(json);
        assertEquals(conf,conf2);
    }
}
