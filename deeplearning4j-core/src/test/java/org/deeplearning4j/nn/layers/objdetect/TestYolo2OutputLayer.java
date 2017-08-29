package org.deeplearning4j.nn.layers.objdetect;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class TestYolo2OutputLayer {

    @Test
    public void testBroadcast45_1(){

        int[] shape4 = new int[]{2,3,4,5};
        int length = ArrayUtil.prod(shape4);

        INDArray fourd = Nd4j.linspace(1, length, length).reshape('c', shape4);

        int[] shape5 = new int[]{2,3,6,4,5};

        // IllegalArgumentException: Incompatible broadcast from [2, 3, 4, 5] to [2, 3, 6, 4, 5]
        // Broadcast [2,3,4,5] to [2,3,6,4,5], should be valid? 6 copies indexed along dimension 2
        INDArray fived = fourd.broadcast(shape5);
        assertArrayEquals(shape5, fived.shape());

        for( int i=0; i<6; i++ ){
            INDArray subset = fived.get(all(), all(), point(i), all(), all());
            assertArrayEquals(shape4, subset.shape());

            assertEquals(fourd, subset);
        }
    }

    @Test
    public void testBroadcast45_2(){

        int[] shape4 = new int[]{3,2,13,13};
        int length = ArrayUtil.prod(shape4);

        INDArray fourd = Nd4j.linspace(1, length, length).reshape('c', shape4);

        int[] shape5 = new int[]{3,3,2,13,13};

        INDArray fived = fourd.broadcast(shape5);
        System.out.println(Arrays.toString(fived.shape())); //Shape [3,3,13,13,13], should be [3,3,2,13,13]
        assertArrayEquals(shape5, fived.shape());

        for( int i=0; i<3; i++ ){
            INDArray subset = fived.get(all(), point(i), all(), all(), all());
            assertArrayEquals(shape4, subset.shape());

            assertEquals(fourd, subset);
        }
    }

    @Test
    public void testBroadcast23_1(){

        int[] shape3 = {2,3,4};

        //[2,3] to [2,3,4]
        INDArray arr23 = Nd4j.linspace(1,6,6).reshape('c', 2,3);
        INDArray bc23 = arr23.broadcast(shape3);    //IllegalArgumentException: Incompatible broadcast from [2, 3] to [2, 3, 4]
        assertArrayEquals(shape3, bc23.shape());

        for( int i=0; i<4; i++ ){
            INDArray sub = bc23.get(all(), all(), point(i));
            assertEquals(arr23, sub);
        }

        //[2,4] to [2,3,4]
        INDArray arr24 = Nd4j.linspace(1,8,8).reshape('c', 2,4);
        INDArray bc24 = arr24.broadcast(shape3);    //IllegalArgumentException: Incompatible broadcast from [2, 4] to [2, 3, 4]
        assertArrayEquals(shape3, bc24.shape());

        for( int i=0; i<3; i++ ){
            INDArray sub = bc24.get(all(), point(i), all());
            assertEquals(arr24, sub);
        }


        //[3,4] to [2,3,4]
        INDArray arr34 = Nd4j.linspace(1,12,12).reshape('c', 3,4);
        INDArray bc34 = arr34.broadcast(shape3);
        assertArrayEquals(shape3, bc34.shape());    //Fails: shape [3,4,4]

        for( int i=0; i<2; i++ ){
            INDArray sub = bc34.get(point(i), all(), all());
            assertEquals(arr34, sub);
        }
    }



    @Test
    public void testYoloActivateScoreBasic(){

        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

        int mb = 3;
        int b = 4;
        int c = 3;
        int depth = 5 * b + c;
        int w = 6;
        int h = 6;

        INDArray bbPrior = Nd4j.rand(b, 2).muliRowVector(Nd4j.create(new double[]{w, h}));


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new ConvolutionLayer.Builder().nIn(1).nOut(1).kernelSize(1,1).build())
                .layer(new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(bbPrior)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer y2impl = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) net.getLayer(1);

        INDArray input = Nd4j.rand(new int[]{mb, depth, h, w});

        INDArray out = y2impl.activate(input);
        assertNotNull(out);
        assertArrayEquals(input.shape(), out.shape());



        //Check score method (simple)
        int labelDepth = 4 + c;
        INDArray labels = Nd4j.zeros(mb, labelDepth, h, w);
        //put 1 object per minibatch, at positions (0,0), (1,1) etc.
        //Positions for label boxes: (1,1) to (2,2), (2,2) to (4,4) etc
        labels.putScalar(0, 4 + 0, 0, 0, 1);
        labels.putScalar(1, 4 + 1, 1, 1, 1);
        labels.putScalar(2, 4 + 2, 2, 2, 1);

        labels.putScalar(0, 0, 0, 0, 1);
        labels.putScalar(0, 1, 0, 0, 1);
        labels.putScalar(0, 2, 0, 0, 2);
        labels.putScalar(0, 3, 0, 0, 2);

        labels.putScalar(1, 0, 1, 1, 2);
        labels.putScalar(1, 1, 1, 1, 2);
        labels.putScalar(1, 2, 1, 1, 4);
        labels.putScalar(1, 3, 1, 1, 4);

        labels.putScalar(2, 0, 2, 2, 3);
        labels.putScalar(2, 1, 2, 2, 3);
        labels.putScalar(2, 2, 2, 2, 6);
        labels.putScalar(2, 3, 2, 2, 6);

        y2impl.setInput(input);
        y2impl.setLabels(labels);
        double score = y2impl.computeScore(0, 0, true);

        System.out.println("SCORE: " + score);
        assertTrue(score > 0.0);
    }

}
