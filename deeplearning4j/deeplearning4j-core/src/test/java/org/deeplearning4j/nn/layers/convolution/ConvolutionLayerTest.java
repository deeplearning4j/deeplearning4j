package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ConvolutionLayerTest extends BaseDL4JTest {

    @Before
    public void before() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.EPS_THRESHOLD = 1e-4;
    }

    @Test
    public void testTwdFirstLayer() throws Exception {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(2e-4)
                        .updater(new Nesterovs(0.9)).dropOut(0.5)
                        .list().layer(0,
                                        new ConvolutionLayer.Builder(8, 8) //16 filters kernel size 8 stride 4
                                                        .stride(4, 4).nOut(16).dropOut(0.5)
                                                        .activation(Activation.RELU).weightInit(
                                                                        WeightInit.XAVIER)
                                                        .build())
                        .layer(1, new ConvolutionLayer.Builder(4, 4) //32 filters kernel size 4 stride 2
                                        .stride(2, 2).nOut(32).dropOut(0.5).activation(Activation.RELU)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(2, new DenseLayer.Builder() //fully connected with 256 rectified units
                                        .nOut(256).activation(Activation.RELU).weightInit(WeightInit.XAVIER)
                                        .dropOut(0.5).build())
                        .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS) //output layer
                                        .nOut(10).weightInit(WeightInit.XAVIER).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutionalFlat(28, 28, 1)).backprop(true).pretrain(false);

        DataSetIterator iter = new MnistDataSetIterator(10, 10);
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        DataSet ds = iter.next();
        for( int i=0; i<5; i++ ) {
            network.fit(ds);
        }
    }

    @Test
    public void testCNNSubComboWithMixedHW() {
        int imageHeight = 20;
        int imageWidth = 23;
        int nChannels = 1;
        int classes = 2;
        int numSamples = 200;

        int kernelHeight = 3;
        int kernelWidth = 3;

        DataSet trainInput;
        MultiLayerConfiguration.Builder builder =
                        new NeuralNetConfiguration.Builder()
                                        .seed(123)
                                        .list()
                                        .layer(0, new ConvolutionLayer.Builder(kernelHeight, kernelWidth).stride(1, 1)
                                                        .nOut(2).activation(Activation.RELU)
                                                        .weightInit(WeightInit.XAVIER).build())
                                        .layer(1, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                                                        .kernelSize(imageHeight - kernelHeight, 1).stride(1, 1).build())
                                        .layer(2, new OutputLayer.Builder().nOut(classes).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).build())
                                        .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, nChannels))
                                        .backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        INDArray emptyFeatures = Nd4j.zeros(numSamples, imageWidth * imageHeight * nChannels);
        INDArray emptyLables = Nd4j.zeros(numSamples, classes);

        trainInput = new DataSet(emptyFeatures, emptyLables);
        model.fit(trainInput);
    }


    @Test(expected = DL4JException.class)
    public void testCNNTooLargeKernel() {
        int imageHeight = 20;
        int imageWidth = 23;
        int nChannels = 1;
        int classes = 2;
        int numSamples = 200;

        int kernelHeight = imageHeight;
        int kernelWidth = imageWidth + 1;

        DataSet trainInput;
        MultiLayerConfiguration.Builder builder =
                        new NeuralNetConfiguration.Builder()
                                        .seed(123)
                                        .list()
                                        .layer(0, new ConvolutionLayer.Builder(kernelHeight, kernelWidth) //(img-kernel+2*padding)/stride + 1: must be >= 1. Therefore: with p=0, kernel <= img size
                                                        .stride(1, 1).nOut(2).activation(Activation.RELU)
                                                        .weightInit(WeightInit.XAVIER).build())
                                        .layer(1, new OutputLayer.Builder().nOut(classes).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).build())
                                        .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, nChannels))
                                        .backprop(true).pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        INDArray emptyFeatures = Nd4j.zeros(numSamples, imageWidth * imageHeight * nChannels);
        INDArray emptyLables = Nd4j.zeros(numSamples, classes);

        trainInput = new DataSet(emptyFeatures, emptyLables);
        model.fit(trainInput);
    }

    @Test(expected = Exception.class)
    public void testCNNZeroStride() {
        int imageHeight = 20;
        int imageWidth = 23;
        int nChannels = 1;
        int classes = 2;
        int numSamples = 200;

        int kernelHeight = imageHeight;
        int kernelWidth = imageWidth;

        DataSet trainInput;
        MultiLayerConfiguration.Builder builder =
                        new NeuralNetConfiguration.Builder()
                                        .seed(123)
                                        .list()
                                        .layer(0, new ConvolutionLayer.Builder(kernelHeight, kernelWidth).stride(1, 0)
                                                        .nOut(2).activation(Activation.RELU)
                                                        .weightInit(WeightInit.XAVIER).build())
                                        .layer(1, new OutputLayer.Builder().nOut(classes).weightInit(WeightInit.XAVIER)
                                                        .activation(Activation.SOFTMAX).build())
                                        .backprop(true).pretrain(false)
                                        .setInputType(InputType.convolutional(imageHeight, imageWidth, nChannels));

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        INDArray emptyFeatures = Nd4j.zeros(numSamples, imageWidth * imageHeight * nChannels);
        INDArray emptyLables = Nd4j.zeros(numSamples, classes);

        trainInput = new DataSet(emptyFeatures, emptyLables);
        model.fit(trainInput);
    }

    @Test(expected = Exception.class)
    @Ignore //Test is based on the assumption that there is no default kernel size. Needs to be revisited.
    public void testCNNKernelNoSize() {
        int imageHeight = 20;
        int imageWidth = 23;
        int nChannels = 1;
        int classes = 2;
        int numSamples = 200;

        int kernelHeight = 3;
        int kernelWidth = 3;

        DataSet trainInput;
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(123).list()
                        .layer(0, new ConvolutionLayer.Builder().nOut(2).activation(Activation.RELU)
                                        .weightInit(WeightInit.XAVIER).build())
                        .layer(1, new SubsamplingLayer.Builder().poolingType(SubsamplingLayer.PoolingType.MAX).build())
                        .layer(2, new OutputLayer.Builder().nOut(classes).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutionalFlat(imageHeight, imageWidth, nChannels)).backprop(true)
                        .pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        INDArray emptyFeatures = Nd4j.zeros(numSamples, imageWidth * imageHeight * nChannels);
        INDArray emptyLables = Nd4j.zeros(numSamples, classes);

        trainInput = new DataSet(emptyFeatures, emptyLables);
        model.fit(trainInput);
    }

    @Test
    public void testCNNBiasInit() {
        ConvolutionLayer cnn = new ConvolutionLayer.Builder().nIn(1).nOut(3).biasInit(1).build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layer(cnn).build();

        val numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        Layer layer = conf.getLayer().instantiate(conf, null, 0, params, true);

        assertEquals(1, layer.getParam("b").size(0));
    }

    @Test
    public void testCNNInputSetupMNIST() throws Exception {
        INDArray input = getMnistData();
        Layer layer = getMNISTConfig();
        layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(input, layer.input());
        assertArrayEquals(input.shape(), layer.input().shape());
    }

    @Test
    public void testFeatureMapShapeMNIST() throws Exception {
        int inputWidth = 28;
        int[] stride = new int[] {1, 1};
        int[] padding = new int[] {0, 0};
        int[] kernelSize = new int[] {9, 9};
        int nChannelsIn = 1;
        int depth = 20;
        int featureMapWidth = (inputWidth + padding[1] * 2 - kernelSize[1]) / stride[1] + 1;

        INDArray input = getMnistData();

        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);
        INDArray convActivations = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        assertEquals(featureMapWidth, convActivations.size(2));
        assertEquals(depth, convActivations.size(1));
    }

    @Test
    public void testActivateResultsContained() {
        Layer layer = getContainedConfig();
        INDArray input = getContainedData();
        INDArray expectedOutput = Nd4j.create(new double[] {0.98201379, 0.98201379, 0.98201379, 0.98201379, 0.99966465,
                        0.99966465, 0.99966465, 0.99966465, 0.98201379, 0.98201379, 0.98201379, 0.98201379, 0.99966465,
                        0.99966465, 0.99966465, 0.99966465, 0.98201379, 0.98201379, 0.98201379, 0.98201379, 0.99966465,
                        0.99966465, 0.99966465, 0.99966465, 0.98201379, 0.98201379, 0.98201379, 0.98201379, 0.99966465,
                        0.99966465, 0.99966465, 0.99966465}, new int[] {1, 2, 4, 4});

        INDArray convActivations = layer.activate(input, false, LayerWorkspaceMgr.noWorkspaces());

        assertArrayEquals(expectedOutput.shape(), convActivations.shape());
        assertEquals(expectedOutput, convActivations);
    }

    //////////////////////////////////////////////////////////////////////////////////

    private static Layer getCNNConfig(int nIn, int nOut, int[] kernelSize, int[] stride, int[] padding) {

        ConvolutionLayer layer = new ConvolutionLayer.Builder(kernelSize, stride, padding).nIn(nIn).nOut(nOut)
                        .activation(Activation.SIGMOID).build();

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder().layer(layer).build();

        val numParams = conf.getLayer().initializer().numParams(conf);
        INDArray params = Nd4j.create(1, numParams);
        return conf.getLayer().instantiate(conf, null, 0, params, true);
    }

    public Layer getMNISTConfig() {
        int[] kernelSize = new int[] {9, 9};
        int[] stride = new int[] {1, 1};
        int[] padding = new int[] {1, 1};
        int nChannelsIn = 1;
        int depth = 20;

        return getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);

    }

    public INDArray getMnistData() throws Exception {
        int inputWidth = 28;
        int inputHeight = 28;
        int nChannelsIn = 1;
        int nExamples = 5;

        DataSetIterator data = new MnistDataSetIterator(nExamples, nExamples);
        DataSet mnist = data.next();
        nExamples = mnist.numExamples();
        return mnist.getFeatureMatrix().reshape(nExamples, nChannelsIn, inputHeight, inputWidth);
    }

    public Layer getContainedConfig() {
        int[] kernelSize = new int[] {2, 2};
        int[] stride = new int[] {2, 2};
        int[] padding = new int[] {0, 0};
        int nChannelsIn = 1;
        int depth = 2;

        INDArray W = Nd4j.create(new double[] {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, new int[] {2, 1, 2, 2});
        INDArray b = Nd4j.create(new double[] {1, 1});
        Layer layer = getCNNConfig(nChannelsIn, depth, kernelSize, stride, padding);
        layer.setParam("W", W);
        layer.setParam("b", b);

        return layer;

    }

    public INDArray getContainedData() {
        INDArray ret = Nd4j.create(new double[] {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
                        4, 4, 4, 4, 4, 4, 4, 4}, new int[] {1, 1, 8, 8});
        return ret;
    }

    public INDArray getContainedCol() {
        return Nd4j.create(new double[] {1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1,
                        1, 1, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2,
                        2, 2, 4, 4, 4, 4}, new int[] {1, 1, 2, 2, 4, 4});
    }



    //////////////////////////////////////////////////////////////////////////////////


    @Test
    public void testCNNMLNPretrain() throws Exception {
        // Note CNN does not do pretrain
        int numSamples = 10;
        int batchSize = 10;
        DataSetIterator mnistIter = new MnistDataSetIterator(batchSize, numSamples, true);

        MultiLayerNetwork model = getCNNMLNConfig(false, true);
        model.fit(mnistIter);

        mnistIter.reset();

        MultiLayerNetwork model2 = getCNNMLNConfig(false, true);
        model2.fit(mnistIter);
        mnistIter.reset();

        DataSet test = mnistIter.next();

        Evaluation eval = new Evaluation();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        double f1Score = eval.f1();

        Evaluation eval2 = new Evaluation();
        INDArray output2 = model2.output(test.getFeatureMatrix());
        eval2.eval(test.getLabels(), output2);
        double f1Score2 = eval2.f1();

        assertEquals(f1Score, f1Score2, 1e-4);


    }


    @Test
    public void testCNNMLNBackprop() throws Exception {
        int numSamples = 10;
        int batchSize = 10;
        DataSetIterator mnistIter = new MnistDataSetIterator(batchSize, numSamples, true);

        MultiLayerNetwork model = getCNNMLNConfig(true, false);
        model.fit(mnistIter);

        MultiLayerNetwork model2 = getCNNMLNConfig(true, false);
        model2.fit(mnistIter);

        DataSet test = mnistIter.next();

        Evaluation eval = new Evaluation();
        INDArray output = model.output(test.getFeatureMatrix());
        eval.eval(test.getLabels(), output);
        double f1Score = eval.f1();

        Evaluation eval2 = new Evaluation();
        INDArray output2 = model2.output(test.getFeatureMatrix());
        eval2.eval(test.getLabels(), output2);
        double f1Score2 = eval2.f1();

        assertEquals(f1Score, f1Score2, 1e-4);

    }

    @Test
    public void testGetSetParams() {

        MultiLayerNetwork net = getCNNMLNConfig(true, false);

        INDArray paramsOrig = net.params().dup();
        net.setParams(paramsOrig);

        INDArray params2 = net.params();

        assertEquals(paramsOrig, params2);
    }

    private static final int kH = 2;
    private static final int kW = 2;
    private static final int[] strides = {1, 1};
    private static final int[] pad = {0, 0};

    private static final int miniBatch = 2;
    private static final int inDepth = 2;
    private static final int height = 3;
    private static final int width = 3;

    private static final int outW = 2;
    private static final int outH = 2;

    private static INDArray getInput() {

        /*
         ----- Input images -----
        example 0:
        channels 0     channels 1
        [ 0  1  2      [ 9 10 11
          3  4  5       12 13 14
          6  7  8]      15 16 17]
        example 1:
        [18 19 20      [27 28 29
         21 22 23       30 31 32
         24 25 26]      33 34 35]
         */

        INDArray input = Nd4j.create(new int[] {miniBatch, inDepth, height, width}, 'c');
        input.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
        input.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
        input.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}));
        input.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{27, 28, 29}, {30, 31, 32}, {33, 34, 35}}));

        return input;
    }

    @Test
    public void testCnnIm2ColReshaping() {
        //This test: a bit unusual in that it tests the *assumptions* of the CNN implementation rather than the implementation itself
        //Specifically, it tests the row and column orders after reshaping on im2col is reshaped (both forward and backward pass)
        INDArray input = getInput();

        //im2col in the required order: want [outW,outH,miniBatch,depthIn,kH,kW], but need to input [miniBatch,channels,kH,kW,outH,outW]
        // given the current im2col implementation
        //To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        //to get old order from required order: permute(2,3,4,5,1,2)
        INDArray col = Nd4j.create(new int[] {miniBatch, outH, outW, inDepth, kH, kW}, 'c');
        INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);
        Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], false, col2);

        /*
        Expected Output, im2col
        - example 0 -
            channels 0                        channels 1
        h0,w0      h0,w1               h0,w0      h0,w1
        0  1     1  2                 9 10      10 11
        3  4     4  5                12 13      13 14
        
        h1,w0      h1,w1               h1,w0      h1,w1
        3  4     4  5                12 13      13 14
        6  7     7  8                15 16      16 17
        
        - example 1 -
            channels 0                        channels 1
        h0,w0      h0,w1               h0,w0      h0,w1
        18 19     19 20               27 28      28 29
        21 22     22 23               30 31      31 32
        
        h1,w0      h1,w1               h1,w0      h1,w1
        21 22     22 23               30 31      31 32
        24 25     25 26               33 34      34 35
        */

        //Now, after reshaping im2col to 2d, we expect:
        //Rows with order (wOut0,hOut0,mb0), (wOut1,hOut0,mb0), (wOut0,hOut1,mb0), (wOut1,hOut1,mb0), (wOut0,hOut0,mb1), ...
        //Columns with order (d0,kh0,kw0), (d0,kh0,kw1), (d0,kh1,kw0), (d0,kh1,kw1), (d1,kh0,kw0), ...

        INDArray reshapedCol = Shape.newShapeNoCopy(col, new int[] {miniBatch * outH * outW, inDepth * kH * kW}, false);

        INDArray exp2d = Nd4j.create(outW * outH * miniBatch, inDepth * kH * kW);
        exp2d.putRow(0, Nd4j.create(new double[] {0, 1, 3, 4, 9, 10, 12, 13})); //wOut0,hOut0,mb0 -> both depths, in order (d0,kh0,kw0), (d0,kh0,kw1), (d0,kh1,kw0), (d0,kh1,kw1), (d1,kh0,kw0), (d1,kh0,kw1), (d1,kh1,kw0), (d1,kh1,kw1)
        exp2d.putRow(1, Nd4j.create(new double[] {1, 2, 4, 5, 10, 11, 13, 14})); //wOut1,hOut0,mb0
        exp2d.putRow(2, Nd4j.create(new double[] {3, 4, 6, 7, 12, 13, 15, 16})); //wOut0,hOut1,mb0
        exp2d.putRow(3, Nd4j.create(new double[] {4, 5, 7, 8, 13, 14, 16, 17})); //wOut1,hOut1,mb0
        exp2d.putRow(4, Nd4j.create(new double[] {18, 19, 21, 22, 27, 28, 30, 31})); //wOut0,hOut0,mb1
        exp2d.putRow(5, Nd4j.create(new double[] {19, 20, 22, 23, 28, 29, 31, 32})); //wOut1,hOut0,mb1
        exp2d.putRow(6, Nd4j.create(new double[] {21, 22, 24, 25, 30, 31, 33, 34})); //wOut0,hOut1,mb1
        exp2d.putRow(7, Nd4j.create(new double[] {22, 23, 25, 26, 31, 32, 34, 35})); //wOut1,hOut1,mb1

        assertEquals(exp2d, reshapedCol);

        //Check the same thing for the backprop im2col (different order)
        INDArray colBackprop = Nd4j.create(new int[] {miniBatch, outH, outW, inDepth, kH, kW}, 'c');
        INDArray colBackprop2 = colBackprop.permute(0, 3, 4, 5, 1, 2);

        Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], false, colBackprop2);

        INDArray reshapedColBackprop = Shape.newShapeNoCopy(colBackprop,
                        new int[] {miniBatch * outH * outW, inDepth * kH * kW}, false);

        //Rows with order (mb0,h0,w0), (mb0,h0,w1), (mb0,h1,w0), (mb0,h1,w1), (mb1,h0,w0), (mb1,h0,w1), (mb1,h1,w0), (mb1,h1,w1)
        //Columns with order (d0,kh0,kw0), (d0,kh0,kw1), (d0,kh1,kw0), (d0,kh1,kw1), (d1,kh0,kw0), ...

        INDArray exp2dv2 = Nd4j.create(outW * outH * miniBatch, inDepth * kH * kW);
        exp2dv2.putRow(0, Nd4j.create(new double[] {0, 1, 3, 4, 9, 10, 12, 13})); //wOut0,hOut0,mb0 -> both depths, in order (d0,kh0,kw0), (d0,kh0,kw1), (d0,kh1,kw0), (d0,kh1,kw1), (d1,kh0,kw0), (d1,kh0,kw1), (d1,kh1,kw0), (d1,kh1,kw1)
        exp2dv2.putRow(1, Nd4j.create(new double[] {1, 2, 4, 5, 10, 11, 13, 14})); //wOut1,hOut0,mb0
        exp2dv2.putRow(2, Nd4j.create(new double[] {3, 4, 6, 7, 12, 13, 15, 16})); //wOut0,hOut1,mb0
        exp2dv2.putRow(3, Nd4j.create(new double[] {4, 5, 7, 8, 13, 14, 16, 17})); //wOut1,hOut1,mb0
        exp2dv2.putRow(4, Nd4j.create(new double[] {18, 19, 21, 22, 27, 28, 30, 31})); //wOut0,hOut0,mb1
        exp2dv2.putRow(5, Nd4j.create(new double[] {19, 20, 22, 23, 28, 29, 31, 32})); //wOut1,hOut0,mb1
        exp2dv2.putRow(6, Nd4j.create(new double[] {21, 22, 24, 25, 30, 31, 33, 34})); //wOut0,hOut1,mb1
        exp2dv2.putRow(7, Nd4j.create(new double[] {22, 23, 25, 26, 31, 32, 34, 35})); //wOut1,hOut1,mb1

        assertEquals(exp2dv2, reshapedColBackprop);
    }

    @Test
    public void testDeltaReshaping() {
        //As per above test: testing assumptions of cnn implementation...

        //Delta: initially shape [miniBatch,dOut,outH,outW]
        //permute to [dOut,miniB,outH,outW]
        //then reshape to [dOut,miniB*outH*outW]
        //Expect columns of delta2d to be like: (mb0,h0,w0), (mb0,h0,w1), (mb1,h0,w2), (mb0,h1,w0), ... (mb1,...), ..., (mb2,...)
        int miniBatch = 3;
        int depth = 2;
        int outW = 3;
        int outH = 3;

        /*
         ----- Input delta -----
        example 0:
        channels 0     channels 1
        [ 0  1  2      [ 9 10 11
          3  4  5       12 13 14
          6  7  8]      15 16 17]
        example 1:
        [18 19 20      [27 28 29
         21 22 23       30 31 32
         24 25 26]      33 34 35]
        example 2:
        [36 37 38      [45 46 47
         39 40 41       48 49 50
         42 43 44]      51 52 53]
         */

        INDArray deltaOrig = Nd4j.create(new int[] {miniBatch, depth, outH, outW}, 'c');
        deltaOrig.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}));
        deltaOrig.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{9, 10, 11}, {12, 13, 14}, {15, 16, 17}}));
        deltaOrig.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{18, 19, 20}, {21, 22, 23}, {24, 25, 26}}));
        deltaOrig.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{27, 28, 29}, {30, 31, 32}, {33, 34, 35}}));
        deltaOrig.put(new INDArrayIndex[] {NDArrayIndex.point(2), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{36, 37, 38}, {39, 40, 41}, {42, 43, 44}}));
        deltaOrig.put(new INDArrayIndex[] {NDArrayIndex.point(2), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{45, 46, 47}, {48, 49, 50}, {51, 52, 53}}));


        INDArray deltaPermute = deltaOrig.permute(1, 0, 2, 3).dup('c');
        INDArray delta2d = Shape.newShapeNoCopy(deltaPermute, new int[] {depth, miniBatch * outW * outH}, false);

        INDArray exp = Nd4j.create(new double[][] {
                        {0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41, 42, 43,
                                        44}, //depth0
                        {9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31, 32, 33, 34, 35, 45, 46, 47, 48, 49, 50,
                                        51, 52, 53} //depth1
        });

        assertEquals(exp, delta2d);
    }

    @Test
    public void testWeightReshaping() {
        //Test assumptions of weight reshaping
        //Weights: originally c order, shape [outDepth, inDepth, kH, kw]
        //permute (3,2,1,0)

        int depthOut = 2;
        int depthIn = 3;
        int kH = 2;
        int kW = 2;

        /*
         ----- Weights -----
         - dOut 0 -
        dIn 0      dIn 1        dIn 2
        [ 0  1      [ 4  5      [ 8  9
          2  3]       6  7]      10 11]
         - dOut 1 -
        [12 13      [16 17      [20 21
         14 15]      18 19]      22 23]
         */

        INDArray weightOrig = Nd4j.create(new int[] {depthOut, depthIn, kH, kW}, 'c');
        weightOrig.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{0, 1}, {2, 3}}));
        weightOrig.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{4, 5}, {6, 7}}));
        weightOrig.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.point(2), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{8, 9}, {10, 11}}));
        weightOrig.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{12, 13}, {14, 15}}));
        weightOrig.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(1), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{16, 17}, {18, 19}}));
        weightOrig.put(new INDArrayIndex[] {NDArrayIndex.point(1), NDArrayIndex.point(2), NDArrayIndex.all(),
                        NDArrayIndex.all()}, Nd4j.create(new double[][] {{20, 21}, {22, 23}}));

        INDArray weightPermute = weightOrig.permute(3, 2, 1, 0);
        INDArray w2d = Shape.newShapeNoCopy(weightPermute, new int[] {depthIn * kH * kW, depthOut}, true);

        assertNotNull(w2d);

        //Expected order of weight rows, after reshaping: (kw0,kh0,din0), (kw1,kh0,din0), (kw0,kh1,din0), (kw1,kh1,din0), (kw0,kh0,din1), ...
        INDArray wExp = Nd4j.create(new double[][] {{0, 12}, {1, 13}, {2, 14}, {3, 15}, {4, 16}, {5, 17}, {6, 18},
                        {7, 19}, {8, 20}, {9, 21}, {10, 22}, {11, 23}});

        assertEquals(wExp, w2d);
    }

    //////////////////////////////////////////////////////////////////////////////////

    private static MultiLayerNetwork getCNNMLNConfig(boolean backprop, boolean pretrain) {
        int outputNum = 10;
        int seed = 123;

        MultiLayerConfiguration.Builder conf =
                        new NeuralNetConfiguration.Builder().seed(seed)
                                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).list()
                                        .layer(0, new ConvolutionLayer.Builder(new int[] {10, 10}).nOut(6).build())
                                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX,
                                                        new int[] {2, 2}).stride(1, 1).build())
                                        .layer(2, new OutputLayer.Builder(
                                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                                                        .nOut(outputNum).weightInit(WeightInit.XAVIER)
                                                                        .activation(Activation.SOFTMAX).build())
                                        .setInputType(InputType.convolutionalFlat(28, 28, 1)).backprop(backprop)
                                        .pretrain(pretrain);

        MultiLayerNetwork model = new MultiLayerNetwork(conf.build());
        model.init();

        return model;
    }
}
