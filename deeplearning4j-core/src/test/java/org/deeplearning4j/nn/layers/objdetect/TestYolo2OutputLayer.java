package org.deeplearning4j.nn.layers.objdetect;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.List;

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
        MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);

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


    @Test
    public void testYoloOverfitting() throws Exception {

        InputStream is1 = new ClassPathResource("yolo/VOC_SingleImage/JPEGImages/2007_009346.jpg").getInputStream();
        InputStream is2 = new ClassPathResource("yolo/VOC_SingleImage/Annotations/2007_009346.xml").getInputStream();

        File dir = Files.createTempDirectory("testYoloOverfitting").toFile();
        File jpg = new File(dir, "JPEGImages");
        File annot = new File(dir, "Annotations");
        jpg.mkdirs();
        annot.mkdirs();

        File imgOut = new File(jpg, "2007_009346.jpg");
        File annotationOut = new File(annot, "2007_009346.xml");

        try(FileOutputStream fos = new FileOutputStream(imgOut)){
            IOUtils.copy(is1, fos);
        } finally {
            is1.close();
        }

        try(FileOutputStream fos = new FileOutputStream(annotationOut)){
            IOUtils.copy(is2, fos);
        } finally {
            is2.close();
        }

        INDArray bbPriors = Nd4j.create(new double[][]{
                {3,3},
                {5,5}});

        //4x downsampling to 13x13 = 52x52 input images
        //Required depth at output layer: 5B+C, with B=5, C=20 object classes, for VOC
        VocLabelProvider lp = new VocLabelProvider(dir.getPath());
        int h = 52;
        int w = 52;
        int c = 3;
        int depthOut = bbPriors.size(0)*5 + 20;
        RecordReader rr = new ObjectDetectionRecordReader(52, 52, 3, 13, 13, lp);
        rr.initialize(new FileSplit(jpg));

        DataSetIterator iter = new RecordReaderDataSetIterator(rr,1,1,1,true);


//        INDArray bbPriors = Nd4j.create(new double[][]{
//                {3,3},
//                {3,5},
//                {5,3},
//                {5,5},
//                {5,7}});

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .convolutionMode(ConvolutionMode.Same)
                .updater(new Adam(0.01))
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nOut(32).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nOut(64).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())
                .layer(new ConvolutionLayer.Builder().activation(Activation.IDENTITY).kernelSize(3,3).stride(1,1).nOut(depthOut).build())
                .layer(new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(bbPriors)
                        .build())
                .setInputType(InputType.convolutional(h,w,c))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        int nEpochs = 200;
        for( int i=0; i<nEpochs; i++ ){
            net.fit(iter);
        }

        iter.reset();
        DataSet ds = iter.next();

        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer ol =
                (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) net.getLayer(5);

        INDArray out = net.output(ds.getFeatures());

        for( int i=0; i<bbPriors.size(0); i++ ) {
            INDArray confidenceEx0 = ol.getConfidenceMatrix(out, 0, i).dup();

            System.out.println(confidenceEx0);
            System.out.println("\n");
        }

        List<DetectedObject> l = ol.getPredictedObjects(out, 0.5);

    }
}
