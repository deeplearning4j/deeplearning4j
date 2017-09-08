package org.deeplearning4j.nn.layers.objdetect;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class TestYolo2OutputLayer {

    @Test
    public void testYoloActivateScoreBasic() throws Exception {

        //Note that we expect some NaNs here - 0/0 for example in IOU calculation. This is handled explicitly in the
        //implementation
        //Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

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


        //Finally: test ser/de:
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ModelSerializer.writeModel(net, baos, true);
        byte[] bytes = baos.toByteArray();
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        MultiLayerNetwork netLoaded = ModelSerializer.restoreMultiLayerNetwork(bais, true);

        assertEquals(net.params(), netLoaded.params());
        assertEquals(net.getLayerWiseConfigurations(), netLoaded.getLayerWiseConfigurations());

        y2impl = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) netLoaded.getLayer(1);
        y2impl.setInput(input);
        y2impl.setLabels(labels);
        double score2 = y2impl.computeScore(0, 0, true);

        assertEquals(score, score2, 1e-8);
    }


    @Test
    public void testYoloActivateSanityCheck(){

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

        assertEquals(4, out.rank());


        //Check values for x/y, confidence: all should be 0 to 1
        INDArray out4 = out.get(all(), interval(0,5*b), all(), all()).dup('c');
        INDArray out5 = out4.reshape(mb, b, 5, h, w);

        INDArray predictedXYCenterGrid = out5.get(all(), all(), interval(0,2), all(), all());
        INDArray predictedWH = out5.get(all(), all(), interval(2,4), all(), all());   //Shape: [mb, B, 2, H, W]
        INDArray predictedConf = out5.get(all(), all(), point(4), all(), all());   //Shape: [mb, B, H, W]


        assertTrue(predictedXYCenterGrid.minNumber().doubleValue() >= 0.0);
        assertTrue(predictedXYCenterGrid.maxNumber().doubleValue() <= 1.0);
        assertTrue(predictedWH.minNumber().doubleValue() >= 0.0);
        assertTrue(predictedConf.minNumber().doubleValue() >= 0.0);
        assertTrue(predictedConf.maxNumber().doubleValue() <= 1.0);


        //Check classes:
        INDArray probs = out.get(all(), interval(5*b, 5*b+c), all(), all());   //Shape: [minibatch, C, H, W]
        assertTrue(probs.minNumber().doubleValue() >= 0.0);
        assertTrue(probs.maxNumber().doubleValue() <= 1.0);

        INDArray probsSum = probs.sum(1);
        assertEquals(1.0, probsSum.minNumber().doubleValue(), 1e-6);
        assertEquals(1.0, probsSum.maxNumber().doubleValue(), 1e-6);
    }
}
