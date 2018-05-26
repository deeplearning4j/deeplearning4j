package org.deeplearning4j.integration.testcases;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.SvhnLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher;
import org.deeplearning4j.integration.TestCase;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping2D;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.*;

public class CNN2DTestCases {

    /**
     * Essentially: LeNet MNIST example
     */
    public static TestCase getLenetMnist(){
        return new TestCase() {
            {
                testName = "LenetMnist";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = true;
                maxRelativeErrorOverfit = 2e-2;
                minAbsErrorOverfit = 1e-2;
            }

            @Override
            public Object getConfiguration() {
                Map<Integer, Double> lrSchedule = new HashMap<>();
                lrSchedule.put(0, 0.01);
                lrSchedule.put(1000, 0.005);
                lrSchedule.put(3000, 0.001);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                .nIn(1)
                                .stride(1, 1)
                                .nOut(20)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                //Note that nIn need not be specified in later layers
                                .stride(1, 1)
                                .nOut(50)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                .nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(10)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                        .backprop(true).pretrain(false).build();

                return conf;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MnistDataSetIterator iter = new MnistDataSetIterator(1, true, 12345);
                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));

                iter = new MnistDataSetIterator(10, true, 12345);
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));
                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds = new MnistDataSetIterator(10, true, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(32, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()
                };
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(32, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {
                DataSet ds = new MnistDataSetIterator(1, true, 12345).next();
                return ComputationGraphUtil.toMultiDataSet(ds);
            }

            @Override
            public int getOverfitNumIterations(){
                return 200;
            }
        };
    }


    /**
     * VGG16 + transfer learning + tiny imagenet
     */
    public static TestCase getVGG16TransferTinyImagenet(){
        return new TestCase() {

            {
                testName = "VGG16TransferTinyImagenet_224";
                testType = TestType.PRETRAINED;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = false;              //Skip - requires saving approx 1GB of data (gradients x2)
                testParamsPostTraining = false;     //Skip - requires saving all params (approx 500mb)
                testEvaluation = true;
                testOverfitting = false;
            }

            @Override
            public Model getPretrainedModel() throws Exception {
                VGG16 vgg16 = VGG16.builder()
                        .seed(12345)
                        .build();

                ComputationGraph pretrained = (ComputationGraph) vgg16.initPretrained(PretrainedType.IMAGENET);

                //Transfer learning
                ComputationGraph newGraph = new TransferLearning.GraphBuilder(pretrained)
                        .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                                .updater(new Adam(1e-3))
                                .seed(12345)
                            .build())
                        .removeVertexKeepConnections("predictions")
                        .addLayer("predictions", new OutputLayer.Builder()
                                .nIn(4096)
                                .nOut(200)  //Tiny imagenet
                                .build(), "fc2")
                        .build();

                return newGraph;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();

                DataSetIterator iter = new TinyImageNetDataSetIterator(1, new int[]{224,224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                DataSet ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                iter = new TinyImageNetDataSetIterator(3, new int[]{224,224}, DataSetType.TRAIN, null, 54321);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds =  new TinyImageNetDataSetIterator(8, new int[]{224,224}, DataSetType.TRAIN, null, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(4, new int[]{224,224}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());

                iter = new EarlyTerminationDataSetIterator(iter, 6);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()
                };
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(4, new int[]{224,224}, DataSetType.TEST, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                iter = new EarlyTerminationDataSetIterator(iter, 6);
                return new MultiDataSetIteratorAdapter(iter);
            }
        };
    }


    /**
     * Basically a cut-down version of the YOLO house numbers example
     */
    public static TestCase getYoloHouseNumbers(){
        return new TestCase() {
            {
                testName = "YoloHouseNumbers";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = false;              //Skip - requires saving large gradients
                testParamsPostTraining = false;     //Skip - requires saving large params
                testEvaluation = false;             //Eval for YOLO models not yet implemented
                testOverfitting = false;
            }

            @Override
            public Model getPretrainedModel() throws Exception {

                // parameters matching the pretrained TinyYOLO model
                int width = 416;
                int height = 416;
                int nChannels = 3;
                int gridWidth = 13;
                int gridHeight = 13;

                // number classes (digits) for the SVHN datasets
                int nClasses = 10;

                // parameters for the Yolo2OutputLayer
                int nBoxes = 5;
                double lambdaNoObj = 0.5;
                double lambdaCoord = 1.0;
                double[][] priorBoxes = {{2, 5}, {2.5, 6}, {3, 7}, {3.5, 8}, {4, 9}};
                double detectionThreshold = 0.5;

                // parameters for the training phase
                int batchSize = 10;
                int nEpochs = 20;
                double learningRate = 1e-4;
                double lrMomentum = 0.9;

                int seed = 123;



                ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
                INDArray priors = Nd4j.create(priorBoxes);

                FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                        .seed(seed)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                        .gradientNormalizationThreshold(1.0)
                        .updater(new Adam.Builder().learningRate(learningRate).build())
                        //.updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                        .l2(0.00001)
                        .activation(Activation.IDENTITY)
                        .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                        .inferenceWorkspaceMode(WorkspaceMode.SEPARATE)
                        .build();

                ComputationGraph model = new TransferLearning.GraphBuilder(pretrained)
                        .fineTuneConfiguration(fineTuneConf)
                        .removeVertexKeepConnections("conv2d_9")
                        .addLayer("convolution2d_9",
                                new ConvolutionLayer.Builder(1,1)
                                        .nIn(1024)
                                        .nOut(nBoxes * (5 + nClasses))
                                        .stride(1,1)
                                        .convolutionMode(ConvolutionMode.Same)
                                        .weightInit(WeightInit.XAVIER)
                                        .activation(Activation.IDENTITY)
                                        .build(),
                                "leaky_re_lu_8")
                        .addLayer("outputs",
                                new Yolo2OutputLayer.Builder()
                                        .lambbaNoObj(lambdaNoObj)
                                        .lambdaCoord(lambdaCoord)
                                        .boundingBoxPriors(priors)
                                        .build(),
                                "convolution2d_9")
                        .setOutputs("outputs")
                        .build();

                return model;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MultiDataSet mds1 = getTrainingData().next(1);
                MultiDataSet mds2 = getTrainingData().next(4);

                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();
                out.add(new Pair<>(mds1.getFeatures(), mds1.getFeaturesMaskArrays()));
                out.add(new Pair<>(mds2.getFeatures(), mds2.getFeaturesMaskArrays()));
                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                return getTrainingData().next(4);
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                int width = 416;
                int height = 416;
                int nChannels = 3;
                int gridWidth = 13;
                int gridHeight = 13;

                SvhnDataFetcher fetcher = new SvhnDataFetcher();
                File trainDir = fetcher.getDataSetPath(DataSetType.TRAIN);
                Random rng = new Random(12345);
                FileSplit trainData = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, rng);
                ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                        gridHeight, gridWidth, new SvhnLabelProvider(trainDir));
                recordReaderTest.initialize(trainData);

                DataSetIterator train = new RecordReaderDataSetIterator(recordReaderTest, 4, 1, 1, true);
                train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

                train = new EarlyTerminationDataSetIterator(train, 10);
                return new MultiDataSetIteratorAdapter(train);
            }
        };
    }


    /**
     * A synthetic 2D CNN that uses all layers:
     * Convolution, Subsampling, Upsampling, Cropping, Depthwise conv, separable conv, deconv, space to batch,
     * space to depth, zero padding, batch norm, LRN
     */
    public static TestCase getCnn2DSynthetic(){

        return new TestCase() {
            private final int IMG_DIM = 256;
            private final int[] OUTPUT_SHAPE = new int[]{-1,1,1,1};   //TODO

            {
                testName = "Cnn2DSynthetic";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = false;             //CNNLossLayer
                testOverfitting = true;
            }

            @Override
            public Object getConfiguration(){

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .activation(Activation.SELU)
                        .weightInit(WeightInit.NORMAL)
                        .updater(new AMSGrad(0.001))
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        .layer(new ConvolutionLayer.Builder().nIn(3).nOut(32).kernelSize(3,3).stride(1,1).build())
                        .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(PoolingType.MAX).build())
                        .layer(new Upsampling2D.Builder().size(2).build())
                        .layer(new ConvolutionLayer.Builder().nOut(32).kernelSize(3,3).stride(1,1).build())
                        .layer(new Cropping2D(1,1,0,0))
                        .layer(new ZeroPaddingLayer(0,0,1,1))
                        .layer(new DepthwiseConvolution2D.Builder().depthMultiplier(3).kernelSize(3,3).stride(1,1).build())
                        .layer(new SeparableConvolution2D.Builder().depthMultiplier(2).kernelSize(3,3).stride(1,1).build())
                        .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(PoolingType.AVG).build())
                        .layer(new Deconvolution2D.Builder().kernelSize(2,2).stride(1,1).build())
                        .layer(new SpaceToDepthLayer.Builder(2, SpaceToDepthLayer.DataFormat.NCHW).build())
                        .layer(new ConvolutionLayer.Builder().nIn(3).nOut(32).kernelSize(3,3).stride(1,1).build())
                        .layer(new CnnLossLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunctions.LossFunction.XENT).build())
                        .setInputType(InputType.convolutional(IMG_DIM, IMG_DIM, 3))
                        .build();

                return conf;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();

                DataSetIterator iter = new TinyImageNetDataSetIterator(1, new int[]{IMG_DIM,IMG_DIM}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                DataSet ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                iter = new TinyImageNetDataSetIterator(3, new int[]{IMG_DIM,IMG_DIM}, DataSetType.TRAIN, null, 54321);
                iter.setPreProcessor(new VGG16ImagePreProcessor());
                ds = iter.next();
                out.add(new Pair<>(new INDArray[]{ds.getFeatures()}, null));

                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds =  new TinyImageNetDataSetIterator(8, new int[]{IMG_DIM,IMG_DIM}, DataSetType.TRAIN, null, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new TinyImageNetDataSetIterator(4, new int[]{IMG_DIM,IMG_DIM}, DataSetType.TRAIN, null, 12345);
                iter.setPreProcessor(new VGG16ImagePreProcessor());

                iter = new EarlyTerminationDataSetIterator(iter, 6);
                MultiDataSetIterator mdsIter = new MultiDataSetIteratorAdapter(iter);
                Nd4j.getRandom().setSeed(12345);
                mdsIter.setPreProcessor(mds -> {
                    mds.setLabels(0, Nd4j.rand(OUTPUT_SHAPE));
                });

                return mdsIter;
            }
        };
    }




    public static TestCase testLenetTransferDropoutRepeatability(){
        return new TestCase() {

            {
                testName = "LenetDropoutRepeatability";
                testType = TestType.PRETRAINED;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testParallelInference = true;
                testEvaluation = true;
                testOverfitting = true;
            }

            @Override
            public Model getPretrainedModel() throws Exception {

                Map<Integer, Double> lrSchedule = new HashMap<>();
                lrSchedule.put(0, 0.01);
                lrSchedule.put(1000, 0.005);
                lrSchedule.put(3000, 0.001);

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9))
                        .list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                .nIn(1)
                                .stride(1, 1)
                                .nOut(20)
                                .activation(Activation.IDENTITY)
                                .build())
                        .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                //Note that nIn need not be specified in later layers
                                .stride(1, 1)
                                .nOut(50)
                                .activation(Activation.IDENTITY)
                                .dropOut(0.5)   //**** Dropout on conv layer
                                .build())
                        .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                                .kernelSize(2,2)
                                .stride(2,2)
                                .build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                                .dropOut(0.5)   //**** Dropout on dense layer
                                .nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(10)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28,28,1)) //See note below
                        .backprop(true).pretrain(false).build();


                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                DataSetIterator iter = new EarlyTerminationDataSetIterator(new MnistDataSetIterator(16, true, 12345), 10);
                net.fit(iter);

                MultiLayerNetwork pretrained = new TransferLearning.Builder(net)
                        .fineTuneConfiguration(
                                FineTuneConfiguration.builder()
                                .updater(new Nesterovs(0.01, 0.9))
                                .seed(98765)
                                .build())
                        .nOutReplace(5, 10, WeightInit.XAVIER)
                        .build();

                return pretrained;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MnistDataSetIterator iter = new MnistDataSetIterator(1, true, 12345);
                List<Pair<INDArray[],INDArray[]>> out = new ArrayList<>();
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));

                iter = new MnistDataSetIterator(10, true, 12345);
                out.add(new Pair<>(new INDArray[]{iter.next().getFeatures()}, null));
                return out;
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                DataSet ds = new MnistDataSetIterator(10, true, 12345).next();
                return new org.nd4j.linalg.dataset.MultiDataSet(ds.getFeatures(), ds.getLabels());
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 32);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public IEvaluation[] getNewEvaluations(){
                return new IEvaluation[]{
                        new Evaluation(),
                        new ROCMultiClass(),
                        new EvaluationCalibration()
                };
            }

            @Override
            public MultiDataSetIterator getEvaluationTestData() throws Exception {
                DataSetIterator iter = new MnistDataSetIterator(16, true, 12345);
                iter = new EarlyTerminationDataSetIterator(iter, 10);
                return new MultiDataSetIteratorAdapter(iter);
            }

            @Override
            public MultiDataSet getOverfittingData() throws Exception {
                DataSet ds = new MnistDataSetIterator(1, true, 12345).next();
                return ComputationGraphUtil.toMultiDataSet(ds);
            }

            @Override
            public int getOverfitNumIterations(){
                return 200;
            }
        };
    }
}
