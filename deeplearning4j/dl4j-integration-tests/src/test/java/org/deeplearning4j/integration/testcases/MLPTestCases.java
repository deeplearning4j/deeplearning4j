package org.deeplearning4j.integration.testcases;

import org.deeplearning4j.integration.TestCase;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.EarlyTerminationDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MultiDataSetIteratorAdapter;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.eval.IEvaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MLPTestCases {


    /**
     * A simple MLP test case using MNIST iterator.
     * Also has LR schedule built-in
     */
    public static TestCase getMLPMnist(){
        return new TestCase() {
            {
                testName = "MLPMnist";
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
                return new NeuralNetConfiguration.Builder()
                        .seed(12345)
                        .updater(new Adam(new MapSchedule.Builder(ScheduleType.ITERATION)
                                .add(0, 5e-2)
                                .add(4, 4e-2)
                                .add(8, 3e-2)
                                .add(12, 2e-2)
                                .add(14, 1e-2)
                                .build()))
                        .l1(1e-3).l2(1e-3)
                        .list()
                        .layer(new DenseLayer.Builder().activation(Activation.TANH).nOut(64).build())
                        .layer(new OutputLayer.Builder().nOut(10)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .build())
                        .setInputType(InputType.convolutionalFlat(28,28,1))
                        .build();
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
                return 300;
            }
        };
    }


    /**
     * A test case that mirrors MLP Moon example using CSVRecordReader
     */
    public static TestCase getMLPMoon(){
        return new TestCase() {
            {
                testName = "MLPMoon";
                testType = TestType.RANDOM_INIT;
                testPredictions = true;
                testTrainingCurves = true;
                testGradients = true;
                testParamsPostTraining = true;
                testEvaluation = true;
                testOverfitting = false;    //Not much point here: very simple training data
            }

            @Override
            public Object getConfiguration() {
                int seed = 123;
                double learningRate = 0.005;

                int numInputs = 2;
                int numOutputs = 2;
                int numHiddenNodes = 20;

                //log.info("Build model....");
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .updater(new Nesterovs(learningRate, 0.9))
                        .list()
                        .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.RELU)
                                .build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .nIn(numHiddenNodes).nOut(numOutputs).build())
                        .pretrain(false).backprop(true).build();
                return conf;
            }

            @Override
            public List<Pair<INDArray[], INDArray[]>> getPredictionsTestData() throws Exception {
                File f = new ClassPathResource("dl4j-integration-tests/data/moon_data_eval.csv").getFile();
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSetIterator testIter = new RecordReaderDataSetIterator(rr,1,0,2);
                INDArray next1 = testIter.next().getFeatures();

                testIter = new RecordReaderDataSetIterator(rr,10,0,2);
                INDArray next10 = testIter.next().getFeatures();

                return Arrays.asList(new Pair<>(new INDArray[]{next1}, null),
                        new Pair<>(new INDArray[]{next10}, null));
            }

            @Override
            public MultiDataSet getGradientsTestData() throws Exception {
                File f = new ClassPathResource("dl4j-integration-tests/data/moon_data_eval.csv").getFile();
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSet ds = new RecordReaderDataSetIterator(rr,5,0,2).next();
                return ComputationGraphUtil.toMultiDataSet(ds);
            }

            @Override
            public MultiDataSetIterator getTrainingData() throws Exception {
                File f = new ClassPathResource("dl4j-integration-tests/data/moon_data_train.csv").getFile();
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,32,0,2);
                return new MultiDataSetIteratorAdapter(trainIter);
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
                File f = new ClassPathResource("dl4j-integration-tests/data/moon_data_eval.csv").getFile();
                RecordReader rr = new CSVRecordReader();
                rr.initialize(new FileSplit(f));
                DataSetIterator testIter = new RecordReaderDataSetIterator(rr,32,0,2);
                return new MultiDataSetIteratorAdapter(testIter);
            }

            @Override
            public int getOverfitNumIterations(){
                return 200;
            }
        };
    }
}
