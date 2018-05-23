package integration.testcases;

import integration.TestCase;
import integration.testcases.misc.CharacterIterator;
import integration.testcases.misc.CompositeMultiDataSetPreProcessor;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.MultiDataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.MultiNormalizerStandardize;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.util.Collections;
import java.util.List;

public class RNNTestCases {

    /**
     * RNN + global pooling + CSV + normalizer
     */
    public static TestCase getRnnCsvSequenceClassificationTestCase1(){
        return new RnnCsvSequenceClassificationTestCase1();
    }

    public static TestCase getRnnCsvSequenceClassificationTestCase2(){
        return new RnnCsvSequenceClassificationTestCase2();
    }

    public static TestCase getRnnCharacterTestCase(){
        return new TestCase() {
            {
                testName = "getRnnCharacterTestCase";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;            //Not much point on this one - it already fits very well...
            }

            private int miniBatchSize = 32;
            private int exampleLength = 1000;


            @Override
            public Object getConfiguration() throws Exception {

                CharacterIterator iter = CharacterIterator.getShakespeareIterator(miniBatchSize,exampleLength);
                int nOut = iter.totalOutcomes();

                int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
                int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters

                return new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .l2(0.001)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new RmsProp(0.1))
                        .list()
                        .layer(0, new LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                                .activation(Activation.TANH).build())
                        .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                                .activation(Activation.TANH).build())
                        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                                .nIn(lstmLayerSize).nOut(nOut).build())
                        .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                        .pretrain(false).backprop(true)
                        .build();
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                MultiDataSet mds = getTrainingData().next();
                return Collections.singletonList(new Pair<>(mds.getFeatures(), mds.getFeaturesMaskArrays()));
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                return getTrainingData().next();
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                DataSetIterator iter = CharacterIterator.getShakespeareIterator(miniBatchSize,exampleLength);
                iter = new EarlyTerminationDataSetIterator(iter, 2);    //3 minibatches, 1000/200 = 5 updates per minibatch
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
                return getTrainingData();
            }
        };
    }

    protected static class RnnCsvSequenceClassificationTestCase1 extends TestCase {
        protected   RnnCsvSequenceClassificationTestCase1(){
            testName = "RnnCsvSequenceClassification1";
            testType = TestType.RANDOM_INIT;
            testPredictions = true;
            testTrainingCurves = true;
            testGradients = true;
            testParamsPostTraining = true;
            testEvaluation = true;
            testOverfitting = false;            //Not much point on this one - it already fits very well...
        }


        protected MultiDataNormalization normalizer;

        protected MultiDataNormalization getNormalizer() throws Exception {
            if(normalizer != null){
                return normalizer;
            }

            normalizer = new MultiNormalizerStandardize();
            normalizer.fit(getTrainingDataUnnormalized());

            return normalizer;
        }

        @Override
        public Object getConfiguration() throws Exception {
            return new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .updater(new Adam(5e-2))
                    .l1(1e-3).l2(1e-3)
                    .list()
                    .layer(0, new LSTM.Builder().activation(Activation.TANH).nOut(10).build())
                    .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                    .layer(new OutputLayer.Builder().nOut(6)
                            .lossFunction(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.recurrent(1))
                    .build();
        }

        @Override
        public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
            MultiDataSet mds = getTrainingData().next();
            return Collections.singletonList(new Pair<>(mds.getFeatures(), mds.getFeaturesMaskArrays()));
        }

        @Override
        public MultiDataSet getGradientsTestData() throws Exception {
            return getTrainingData().next();
        }

        @Override
        public MultiDataSetIterator getTrainingData() throws Exception {
            MultiDataSetIterator iter = getTrainingDataUnnormalized();

            MultiDataSetPreProcessor pp = multiDataSet -> {
                INDArray l = multiDataSet.getLabels(0);
                l = l.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(l.size(2)-1));
                multiDataSet.setLabels(0, l);
                multiDataSet.setLabelsMaskArray(0, null);
            };


            iter.setPreProcessor(new CompositeMultiDataSetPreProcessor(getNormalizer(),pp));

            return iter;
        }

        protected MultiDataSetIterator getTrainingDataUnnormalized() throws Exception {
            int miniBatchSize = 10;
            int numLabelClasses = 6;

            File featuresDirTrain = new ClassPathResource("/RnnCsvSequenceClassification/uci/train/features/").getFile();
            File labelsDirTrain = new ClassPathResource("/RnnCsvSequenceClassification/uci/train/labels/").getFile();

            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

            DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                    false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            MultiDataSetIterator iter = new MultiDataSetIteratorAdapter(trainData);
            return iter;
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
            int miniBatchSize = 10;
            int numLabelClasses = 6;

            File featuresDirTest = new ClassPathResource("/RnnCsvSequenceClassification/uci/test/features/").getFile();
            File labelsDirTest = new ClassPathResource("/RnnCsvSequenceClassification/uci/test/labels/").getFile();

            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

            DataSetIterator testData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                    false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            MultiDataSetIterator iter = new MultiDataSetIteratorAdapter(testData);

            MultiDataSetPreProcessor pp = multiDataSet -> {
                INDArray l = multiDataSet.getLabels(0);
                l = l.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(l.size(2)-1));
                multiDataSet.setLabels(0, l);
                multiDataSet.setLabelsMaskArray(0, null);
            };


            iter.setPreProcessor(new CompositeMultiDataSetPreProcessor(getNormalizer(),pp));

            return iter;
        }
    }

    /**
     * Similar to test case 1 - but using GravesLSTM + bidirectional wrapper + min/max scaler normalizer
     */
    protected static class RnnCsvSequenceClassificationTestCase2 extends RnnCsvSequenceClassificationTestCase1 {
        protected RnnCsvSequenceClassificationTestCase2() {
            super();
            testName = "RnnCsvSequenceClassification2";
        }

        @Override
        public Object getConfiguration() throws Exception {
            return new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .updater(new Adam(5e-2))
                    .l1(1e-3).l2(1e-3)
                    .list()
                    .layer(0, new Bidirectional(new LSTM.Builder().activation(Activation.TANH).nOut(10).build()))
                    .layer(new GlobalPoolingLayer.Builder().poolingType(PoolingType.AVG).build())
                    .layer(new OutputLayer.Builder().nOut(6)
                            .lossFunction(LossFunctions.LossFunction.MCXENT)
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.recurrent(1))
                    .build();
        }

        protected MultiDataNormalization getNormalizer() throws Exception {
            if(normalizer != null){
                return normalizer;
            }

            normalizer = new MultiNormalizerMinMaxScaler();
            normalizer.fit(getTrainingDataUnnormalized());

            return normalizer;
        }
    }


}
