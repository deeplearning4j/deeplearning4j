package org.deeplearning4j.nn.layers.objdetect;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.BaseDL4JTest;
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
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URI;
import java.nio.file.Files;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.junit.Assert.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class TestYolo2OutputLayer extends BaseDL4JTest {

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    @Test
    public void testYoloActivateScoreBasic() {

        //Note that we expect some NaNs here - 0/0 for example in IOU calculation. This is handled explicitly in the
        //implementation
        //Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);

        int mb = 3;
        int b = 4;
        int c = 3;
        int depth = b * (5 + c);
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

        INDArray out = y2impl.activate(input, false, LayerWorkspaceMgr.noWorkspaces());
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

        y2impl.setInput(input, LayerWorkspaceMgr.noWorkspaces());
        y2impl.setLabels(labels);
        double score = y2impl.computeScore(0, 0, true, LayerWorkspaceMgr.noWorkspaces());

        System.out.println("SCORE: " + score);
        assertTrue(score > 0.0);


        //Finally: test ser/de:
        MultiLayerNetwork netLoaded = TestUtils.testModelSerialization(net);

        y2impl = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) netLoaded.getLayer(1);
        y2impl.setInput(input, LayerWorkspaceMgr.noWorkspaces());
        y2impl.setLabels(labels);
        double score2 = y2impl.computeScore(0, 0, true, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(score, score2, 1e-8);
    }


    @Test
    public void testYoloActivateSanityCheck(){

        int mb = 3;
        int b = 4;
        int c = 3;
        int depth = b * (5 + c);
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

        INDArray out = y2impl.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(4, out.rank());


        //Check values for x/y, confidence: all should be 0 to 1
        INDArray out5 = out.reshape('c', mb, b, 5+c, h, w);

        INDArray predictedXYCenterGrid = out5.get(all(), all(), interval(0,2), all(), all());
        INDArray predictedWH = out5.get(all(), all(), interval(2,4), all(), all());   //Shape: [mb, B, 2, H, W]
        INDArray predictedConf = out5.get(all(), all(), point(4), all(), all());   //Shape: [mb, B, H, W]


        assertTrue(predictedXYCenterGrid.minNumber().doubleValue() >= 0.0);
        assertTrue(predictedXYCenterGrid.maxNumber().doubleValue() <= 1.0);
        assertTrue(predictedWH.minNumber().doubleValue() >= 0.0);
        assertTrue(predictedConf.minNumber().doubleValue() >= 0.0);
        assertTrue(predictedConf.maxNumber().doubleValue() <= 1.0);


        //Check classes:
        INDArray probs = out5.get(all(), all(), interval(5, 5+c), all(), all());   //Shape: [minibatch, C, H, W]
        assertTrue(probs.minNumber().doubleValue() >= 0.0);
        assertTrue(probs.maxNumber().doubleValue() <= 1.0);

        INDArray probsSum = probs.sum(2);
        assertEquals(1.0, probsSum.minNumber().doubleValue(), 1e-6);
        assertEquals(1.0, probsSum.maxNumber().doubleValue(), 1e-6);
    }

    @Test
    public void testIOUCalc() throws Exception {

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

//        INDArray bbPriors = Nd4j.create(new double[][]{
//                {3, 3},
//                {5, 4}});
        INDArray bbPriors = Nd4j.create(new double[][]{
                {3, 3}});

        VocLabelProvider lp = new VocLabelProvider(dir.getPath());
        int c = 20;
        int depthOut = bbPriors.size(0) * (bbPriors.size(0) + c);

        int origW = 500;
        int origH = 375;
        int inputW = 52;
        int inputH = 52;

        int gridW = 13;
        int gridH = 13;

        RecordReader rr = new ObjectDetectionRecordReader(inputH, inputW, 3, gridH, gridW, lp);
        rr.initialize(new FileSplit(jpg));

        DataSetIterator iter = new RecordReaderDataSetIterator(rr,1,1,1,true);

        //2 objects here:
        //(60,123) to (220,305)
        //(243,105) to (437,317)

        double cx1 = (60+220)/2.0;
        double cy1 = (123+305)/2.0;
        int gridNumX1 = (int)(gridW * cx1 / origW);
        int gridNumY1 = (int)(gridH * cy1 / origH);

        double labelGridBoxX1_tl = gridW * 60.0 / origW;
        double labelGridBoxY1_tl = gridH * 123.0 / origH;
        double labelGridBoxX1_br = gridW * 220.0 / origW;
        double labelGridBoxY1_br = gridH * 305.0 / origH;


        double cx2 = (243+437)/2.0;
        double cy2 = (105+317)/2.0;
        int gridNumX2 = (int)(gridW * cx2 / origW);
        int gridNumY2 = (int)(gridH * cy2 / origH);

        double labelGridBoxX2_tl = gridW * 243.0 / origW;
        double labelGridBoxY2_tl = gridH * 105.0 / origH;
        double labelGridBoxX2_br = gridW * 437.0 / origW;
        double labelGridBoxY2_br = gridH * 317.0 / origH;

        //Check labels
        DataSet ds = iter.next();
        INDArray labelImgClasses = ds.getLabels().get(point(0), point(4), all(), all());
        INDArray labelX_tl = ds.getLabels().get(point(0), point(0), all(), all());
        INDArray labelY_tl = ds.getLabels().get(point(0), point(1), all(), all());
        INDArray labelX_br = ds.getLabels().get(point(0), point(2), all(), all());
        INDArray labelY_br = ds.getLabels().get(point(0), point(3), all(), all());

        INDArray expLabelImg = Nd4j.create(gridH,gridW);
        expLabelImg.putScalar(gridNumY1, gridNumX1, 1.0);
        expLabelImg.putScalar(gridNumY2, gridNumX2, 1.0);

        INDArray expX_TL = Nd4j.create(gridH, gridW);
        expX_TL.putScalar(gridNumY1, gridNumX1, labelGridBoxX1_tl);
        expX_TL.putScalar(gridNumY2, gridNumX2, labelGridBoxX2_tl);

        INDArray expY_TL = Nd4j.create(gridH, gridW);
        expY_TL.putScalar(gridNumY1, gridNumX1, labelGridBoxY1_tl);
        expY_TL.putScalar(gridNumY2, gridNumX2, labelGridBoxY2_tl);

        INDArray expX_BR = Nd4j.create(gridH, gridW);
        expX_BR.putScalar(gridNumY1, gridNumX1, labelGridBoxX1_br);
        expX_BR.putScalar(gridNumY2, gridNumX2, labelGridBoxX2_br);

        INDArray expY_BR = Nd4j.create(gridH, gridW);
        expY_BR.putScalar(gridNumY1, gridNumX1, labelGridBoxY1_br);
        expY_BR.putScalar(gridNumY2, gridNumX2, labelGridBoxY2_br);


        assertEquals(expLabelImg, labelImgClasses);
        assertEquals(expX_TL, labelX_tl);
        assertEquals(expY_TL, labelY_tl);
        assertEquals(expX_BR, labelX_br);
        assertEquals(expY_BR, labelY_br);


        //Check IOU calculation

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nIn(3).nOut(3).build())
                .layer(new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(bbPriors)
                        .build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer ol = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) net.getLayer(1);

        Method m = ol.getClass().getDeclaredMethod("calculateIOULabelPredicted", INDArray.class, INDArray.class, INDArray.class, INDArray.class, INDArray.class);
        m.setAccessible(true);

        INDArray labelTL = ds.getLabels().get(interval(0,1), interval(0,2), all(), all());
        INDArray labelBR = ds.getLabels().get(interval(0,1), interval(2,4), all(), all());

        double pw1 = 2.5;
        double ph1 = 3.5;
        double pw2 = 4.5;
        double ph2 = 5.5;
        INDArray predictedWH = Nd4j.create(1, bbPriors.size(0), 2, gridH, gridW);
        predictedWH.putScalar(new int[]{0, 0, 0, gridNumY1, gridNumX1}, pw1);
        predictedWH.putScalar(new int[]{0, 0, 1, gridNumY1, gridNumX1}, ph1);
        predictedWH.putScalar(new int[]{0, 0, 0, gridNumY2, gridNumX2}, pw2);
        predictedWH.putScalar(new int[]{0, 0, 1, gridNumY2, gridNumX2}, ph2);

        double pX1 = 0.6;
        double pY1 = 0.8;
        double pX2 = 0.3;
        double pY2 = 0.4;
        INDArray predictedXYInGrid = Nd4j.create(1, bbPriors.size(0), 2, gridH, gridW);
        predictedXYInGrid.putScalar(new int[]{0, 0, 0, gridNumY1, gridNumX1}, pX1);
        predictedXYInGrid.putScalar(new int[]{0, 0, 1, gridNumY1, gridNumX1}, pY1);
        predictedXYInGrid.putScalar(new int[]{0, 0, 0, gridNumY2, gridNumX2}, pX2);
        predictedXYInGrid.putScalar(new int[]{0, 0, 1, gridNumY2, gridNumX2}, pY2);

        INDArray objectPresentMask = labelImgClasses;   //Only 1 class here, so same thing as object present mask...

        Object ret = m.invoke(ol, labelTL, labelBR, predictedWH, predictedXYInGrid, objectPresentMask);
        Field fIou = ret.getClass().getDeclaredField("iou");
        fIou.setAccessible(true);
        INDArray iou = (INDArray)fIou.get(ret);


        //Calculate IOU for first image object, first BB
        double predictedTL_x1 = gridNumX1 + pX1 - 0.5 * pw1;
        double predictedTL_y1 = gridNumY1 + pY1 - 0.5 * ph1;
        double predictedBR_x1 = gridNumX1 + pX1 + 0.5 * pw1;
        double predictedBR_y1 = gridNumY1 + pY1 + 0.5 * ph1;

        double intersectionX_TL_1 = Math.max(predictedTL_x1, labelGridBoxX1_tl);
        double intersectionY_TL_1 = Math.max(predictedTL_y1, labelGridBoxY1_tl);
        double intersectionX_BR_1 = Math.min(predictedBR_x1, labelGridBoxX1_br);
        double intersectionY_BR_1 = Math.min(predictedBR_y1, labelGridBoxY1_br);

        double intersection1_bb1 = (intersectionX_BR_1 - intersectionX_TL_1) * (intersectionY_BR_1 - intersectionY_TL_1);
        double pArea1 = pw1 * ph1;
        double lArea1 = (labelGridBoxX1_br - labelGridBoxX1_tl) * (labelGridBoxY1_br - labelGridBoxY1_tl);
        double unionA1 = pArea1 + lArea1 - intersection1_bb1;
        double iou1 = intersection1_bb1 / unionA1;

        //Calculate IOU for second image object, first BB
        double predictedTL_x2 = gridNumX2 + pX2 - 0.5 * pw2;
        double predictedTL_y2 = gridNumY2 + pY2 - 0.5 * ph2;
        double predictedBR_x2 = gridNumX2 + pX2 + 0.5 * pw2;
        double predictedBR_y2 = gridNumY2 + pY2 + 0.5 * ph2;

        double intersectionX_TL_2 = Math.max(predictedTL_x2, labelGridBoxX2_tl);
        double intersectionY_TL_2 = Math.max(predictedTL_y2, labelGridBoxY2_tl);
        double intersectionX_BR_2 = Math.min(predictedBR_x2, labelGridBoxX2_br);
        double intersectionY_BR_2 = Math.min(predictedBR_y2, labelGridBoxY2_br);

        double intersection1_bb2 = (intersectionX_BR_2 - intersectionX_TL_2) * (intersectionY_BR_2 - intersectionY_TL_2);
        double pArea2 = pw2 * ph2;
        double lArea2 = (labelGridBoxX2_br - labelGridBoxX2_tl) * (labelGridBoxY2_br - labelGridBoxY2_tl);
        double unionA2 = pArea2 + lArea2 - intersection1_bb2;
        double iou2 = intersection1_bb2 / unionA2;

        INDArray expIOU = Nd4j.create(1, bbPriors.size(0), gridH, gridW );
        expIOU.putScalar(new int[]{0, 0, gridNumY1, gridNumX1}, iou1);
        expIOU.putScalar(new int[]{0, 0, gridNumY2, gridNumX2}, iou2);

        assertEquals(expIOU, iou);
    }


    @Test
    public void testYoloOverfitting() throws Exception {
        Nd4j.getRandom().setSeed(12345);

        InputStream is1 = new ClassPathResource("yolo/VOC_TwoImage/JPEGImages/2007_009346.jpg").getInputStream();
        InputStream is2 = new ClassPathResource("yolo/VOC_TwoImage/Annotations/2007_009346.xml").getInputStream();
        InputStream is3 = new ClassPathResource("yolo/VOC_TwoImage/JPEGImages/2008_003344.jpg").getInputStream();
        InputStream is4 = new ClassPathResource("yolo/VOC_TwoImage/Annotations/2008_003344.xml").getInputStream();

        File dir = tempDir.newFolder();
        File jpg = new File(dir, "JPEGImages");
        File annot = new File(dir, "Annotations");
        jpg.mkdirs();
        annot.mkdirs();

        File imgOut = new File(jpg, "2007_009346.jpg");
        File annotationOut = new File(annot, "2007_009346.xml");

        try(FileOutputStream fos = new FileOutputStream(imgOut)){
            IOUtils.copy(is1, fos);
        } finally { is1.close(); }

        try(FileOutputStream fos = new FileOutputStream(annotationOut)){
            IOUtils.copy(is2, fos);
        } finally { is2.close(); }

        imgOut = new File(jpg, "2008_003344.jpg");
        annotationOut = new File(annot, "2008_003344.xml");
        try(FileOutputStream fos = new FileOutputStream(imgOut)){
            IOUtils.copy(is3, fos);
        } finally { is3.close(); }

        try(FileOutputStream fos = new FileOutputStream(annotationOut)){
            IOUtils.copy(is4, fos);
        } finally { is4.close(); }

        assertEquals(2, jpg.listFiles().length);
        assertEquals(2, annot.listFiles().length);

        INDArray bbPriors = Nd4j.create(new double[][]{
                {2,2},
                {5,5}});

        //4x downsampling to 13x13 = 52x52 input images
        //Required channels at output layer: 5B+C, with B=5, C=20 object classes, for VOC
        VocLabelProvider lp = new VocLabelProvider(dir.getPath());
        int h = 52;
        int w = 52;
        int c = 3;
        int origW = 500;
        int origH = 375;
        int gridW = 13;
        int gridH = 13;

        RecordReader rr = new ObjectDetectionRecordReader(52, 52, 3, gridH, gridW, lp);
        FileSplit fileSplit = new FileSplit(jpg);
        rr.initialize(fileSplit);

        int nClasses = rr.getLabels().size();
        int depthOut = bbPriors.size(0) * (5 + nClasses);
        // make sure idxCat is not 0 to test DetectedObject.getPredictedClass()
        List<String> labels = rr.getLabels();
        labels.add(labels.remove(labels.indexOf("cat")));
        int idxCat = labels.size() - 1;


        DataSetIterator iter = new RecordReaderDataSetIterator(rr,1,1,1,true);
        iter.setPreProcessor(new ImagePreProcessingScaler());

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .convolutionMode(ConvolutionMode.Same)
                .updater(new Adam(1e-3))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(3)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.RELU)
                .seed(12345)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(5,5).stride(2,2).nOut(256).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2)/*.poolingType(SubsamplingLayer.PoolingType.AVG)*/.build())
                .layer(new ConvolutionLayer.Builder().activation(Activation.IDENTITY).kernelSize(5,5).stride(1,1).nOut(depthOut).build())
                .layer(new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(bbPriors)
                        .build())
                .setInputType(InputType.convolutional(h,w,c))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        int nEpochs = 1000;
        DataSet ds = iter.next();
        URI[] uris = fileSplit.locations();
        if (!uris[0].getPath().contains("2007_009346")) {
            // make sure to get the cat image
            ds = iter.next();
        }
        assertEquals(1, ds.getFeatures().size(0));
        for( int i=0; i<=nEpochs; i++ ){
            net.fit(ds);
        }

        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer ol =
                (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer) net.getLayer(3);

        INDArray out = net.output(ds.getFeatures());

//        for( int i=0; i<bbPriors.size(0); i++ ) {
//            INDArray confidenceEx0 = ol.getConfidenceMatrix(out, 0, i).dup();
//
//            System.out.println(confidenceEx0);
//            System.out.println("\n");
//        }

        List<DetectedObject> l = ol.getPredictedObjects(out, 0.5);
        System.out.println("Detected objects: " + l.size());
        for(DetectedObject d : l){
            System.out.println(d);
        }


        assertEquals(2, l.size());

        //Expect 2 detected objects:
        //(60,123) to (220,305)
        //(243,105) to (437,317)

        double cx1Pixels = (60+220)/2.0;
        double cy1Pixels = (123+305)/2.0;
        double cx1 = gridW * cx1Pixels / origW;
        double cy1 = gridH * cy1Pixels / origH;
        double wGrid1 = (220.0-60.0)/origW * gridW;
        double hGrid1 = (305.0-123.0)/origH * gridH;

        double cx2Pixels = (243+437)/2.0;
        double cy2Pixels = (105+317)/2.0;
        double cx2 = gridW * cx2Pixels / origW;
        double cy2 = gridH * cy2Pixels / origH;
        double wGrid2 = (437.0-243.0)/origW * gridW;
        double hGrid2 = (317-105.0)/origH * gridH;


        //Sort by X position...
        Collections.sort(l, new Comparator<DetectedObject>() {
            @Override
            public int compare(DetectedObject o1, DetectedObject o2) {
                return Double.compare(o1.getCenterX(), o2.getCenterX());
            }
        });


        DetectedObject o1 = l.get(0);
        double p1 = o1.getClassPredictions().getDouble(idxCat);
        double c1 = o1.getConfidence();
        assertEquals(idxCat, o1.getPredictedClass() );
        assertTrue(String.valueOf(p1), p1 >= 0.85);
        assertTrue(String.valueOf(c1), c1 >= 0.85);
        assertEquals(cx1, o1.getCenterX(), 0.1);
        assertEquals(cy1, o1.getCenterY(), 0.1);
        assertEquals(wGrid1, o1.getWidth(), 0.2);
        assertEquals(hGrid1, o1.getHeight(), 0.2);


        DetectedObject o2 = l.get(1);
        double p2 = o2.getClassPredictions().getDouble(idxCat);
        double c2 = o2.getConfidence();
        assertEquals(idxCat, o2.getPredictedClass() );
        assertTrue(String.valueOf(p2), p2 >= 0.85);
        assertTrue(String.valueOf(c2), c2 >= 0.85);
        assertEquals(cx2, o2.getCenterX(), 0.1);
        assertEquals(cy2, o2.getCenterY(), 0.1);
        assertEquals(wGrid2, o2.getWidth(), 0.2);
        assertEquals(hGrid2, o2.getHeight(), 0.2);
    }
}
