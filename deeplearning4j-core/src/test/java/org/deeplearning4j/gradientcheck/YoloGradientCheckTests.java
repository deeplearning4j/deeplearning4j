package org.deeplearning4j.gradientcheck;

import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.List;

import static org.junit.Assert.assertTrue;

/**
 * @author Alex Black
 */
public class YoloGradientCheckTests extends BaseDL4JTest {
    private static final boolean PRINT_RESULTS = true;
    private static final boolean RETURN_ON_FIRST_FAILURE = false;
    private static final double DEFAULT_EPS = 1e-6;
    private static final double DEFAULT_MAX_REL_ERROR = 1e-3;
    private static final double DEFAULT_MIN_ABS_ERROR = 1e-8;

    static {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testYoloOutputLayer() {
        int depthIn = 2;
        int[] minibatchSizes = {1, 3};
        int[] widths = new int[]{4, 7};
        int[] heights = new int[]{4, 5};
        int c = 3;
        int b = 3;

        int yoloDepth = b * (5 + c);
        Activation a = Activation.TANH;

        Nd4j.getRandom().setSeed(1234567);

        double[] l1 = new double[]{0.0, 0.3};
        double[] l2 = new double[]{0.0, 0.4};

        for( int wh = 0; wh<widths.length; wh++ ) {

            int w = widths[wh];
            int h = heights[wh];

            Nd4j.getRandom().setSeed(12345);
            INDArray bbPrior = Nd4j.rand(b, 2).muliRowVector(Nd4j.create(new double[]{w, h})).addi(0.1);

            for (int mb : minibatchSizes) {
                for (int i = 0; i < l1.length; i++) {

                    Nd4j.getRandom().setSeed(12345);

                    INDArray input = Nd4j.rand(new int[]{mb, depthIn, h, w});
                    INDArray labels = yoloLabels(mb, c, h, w);

                    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                            .updater(new NoOp())
                            .activation(a)
                            .l1(l1[i]).l2(l2[i])
                            .convolutionMode(ConvolutionMode.Same)
                            .list()
                            .layer(new ConvolutionLayer.Builder().kernelSize(2, 2).stride(1, 1)
                                    .nIn(depthIn).nOut(yoloDepth).build())//output: (5-2+0)/1+1 = 4
                            .layer(new Yolo2OutputLayer.Builder()
                                    .boundingBoxPriors(bbPrior)
                                    .build())
                            .build();

                    MultiLayerNetwork net = new MultiLayerNetwork(conf);
                    net.init();

                    String msg = "testYoloOutputLayer() - minibatch = " + mb + ", w=" + w + ", h=" + h + ", l1=" + l1[i] + ", l2=" + l2[i];
                    System.out.println(msg);

                    boolean gradOK = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                            DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, input, labels);

                    assertTrue(msg, gradOK);
                }
            }
        }
    }

    private static INDArray yoloLabels(int mb, int c, int h, int w){
        int labelDepth = 4 + c;
        INDArray labels = Nd4j.zeros(mb, labelDepth, h, w);
        //put 1 object per minibatch, at positions (0,0), (1,1) etc.
        //Positions for label boxes: (1,1) to (2,2), (2,2) to (4,4) etc

        for( int i=0; i<mb; i++ ){
            //Class labels
            labels.putScalar(i, 4 + i%c, i%h, i%w, 1);

            //BB coordinates (top left, bottom right)
            labels.putScalar(i, 0, 0, 0, i%w);
            labels.putScalar(i, 1, 0, 0, i%h);
            labels.putScalar(i, 2, 0, 0, (i%w)+1);
            labels.putScalar(i, 3, 0, 0, (i%h)+1);
        }

        return labels;
    }


    @Test
    public void yoloGradientCheckRealData() throws Exception {

        InputStream is1 = new ClassPathResource("yolo/VOC_TwoImage/JPEGImages/2007_009346.jpg").getInputStream();
        InputStream is2 = new ClassPathResource("yolo/VOC_TwoImage/Annotations/2007_009346.xml").getInputStream();
        InputStream is3 = new ClassPathResource("yolo/VOC_TwoImage/JPEGImages/2008_003344.jpg").getInputStream();
        InputStream is4 = new ClassPathResource("yolo/VOC_TwoImage/Annotations/2008_003344.xml").getInputStream();

        File dir = testDir.newFolder("testYoloOverfitting");
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

        INDArray bbPriors = Nd4j.create(new double[][]{
                {3,3},
                {3,5},
                {5,7}});

        //2x downsampling to 13x13 = 26x26 input images
        //Required depth at output layer: 5B+C, with B=3, C=20 object classes, for VOC
        VocLabelProvider lp = new VocLabelProvider(dir.getPath());
        int h = 26;
        int w = 26;
        int c = 3;
        RecordReader rr = new ObjectDetectionRecordReader(h, w, c, 13, 13, lp);
        rr.initialize(new FileSplit(jpg));

        int nClasses = rr.getLabels().size();
        int depthOut = bbPriors.size(0) * (5 + nClasses);


        DataSetIterator iter = new RecordReaderDataSetIterator(rr,2,1,1,true);
        iter.setPreProcessor(new ImagePreProcessingScaler());

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .convolutionMode(ConvolutionMode.Same)
                .updater(new NoOp())
                .weightInit(WeightInit.DISTRIBUTION).dist(new GaussianDistribution(0,0.1))
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nOut(4).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).build())
                .layer(new ConvolutionLayer.Builder().activation(Activation.IDENTITY).kernelSize(3,3).stride(1,1).nOut(depthOut).build())
                .layer(new Yolo2OutputLayer.Builder()
                        .boundingBoxPriors(bbPriors)
                        .build())
                .setInputType(InputType.convolutional(h,w,c))
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        DataSet ds = iter.next();
        INDArray f = ds.getFeatures();
        INDArray l = ds.getLabels();

        boolean ok = GradientCheckUtil.checkGradients(net, DEFAULT_EPS, DEFAULT_MAX_REL_ERROR,
                DEFAULT_MIN_ABS_ERROR, PRINT_RESULTS, RETURN_ON_FIRST_FAILURE, f, l);

        assertTrue(ok);
    }
}
