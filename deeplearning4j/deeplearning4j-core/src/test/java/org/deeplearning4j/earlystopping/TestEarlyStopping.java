package org.deeplearning4j.earlystopping;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.deeplearning4j.earlystopping.scorecalc.AutoencoderScoreCalculator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.scorecalc.RegressionScoreCalculator;
import org.deeplearning4j.earlystopping.scorecalc.VAEReconErrorScoreCalculator;
import org.deeplearning4j.earlystopping.scorecalc.VAEReconProbScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.solvers.BaseOptimizer;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

@Slf4j
public class TestEarlyStopping extends BaseDL4JTest {

    @Test
    public void testEarlyStoppingIris() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(0.001)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                                        .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf, net, irisIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        System.out.println(result);

        assertEquals(5, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition, result.getTerminationReason());
        Map<Integer, Double> scoreVsIter = result.getScoreVsEpoch();
        assertEquals(5, scoreVsIter.size());
        String expDetails = esConf.getEpochTerminationConditions().get(0).toString();
        assertEquals(expDetails, result.getTerminationDetails());

        MultiLayerNetwork out = result.getBestModel();
        assertNotNull(out);

        //Check that best score actually matches (returned model vs. manually calculated score)
        MultiLayerNetwork bestNetwork = result.getBestModel();
        irisIter.reset();
        double score = bestNetwork.score(irisIter.next());
        assertEquals(result.getBestModelScore(), score, 1e-2);
    }

    @Test
    public void testEarlyStoppingEveryNEpoch() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(0.01)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true))
                                        .evaluateEveryNEpochs(2).modelSaver(saver).build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf, net, irisIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        System.out.println(result);

        assertEquals(5, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition, result.getTerminationReason());
    }

    @Test
    public void testEarlyStoppingIrisMultiEpoch() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(0.001)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        MultipleEpochsIterator mIter = new MultipleEpochsIterator(10, irisIter);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                                        .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf, net, mIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        System.out.println(result);

        assertEquals(5, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition, result.getTerminationReason());
        Map<Integer, Double> scoreVsIter = result.getScoreVsEpoch();
        assertEquals(5, scoreVsIter.size());
        String expDetails = esConf.getEpochTerminationConditions().get(0).toString();
        assertEquals(expDetails, result.getTerminationDetails());

        MultiLayerNetwork out = result.getBestModel();
        assertNotNull(out);

        //Check that best score actually matches (returned model vs. manually calculated score)
        MultiLayerNetwork bestNetwork = result.getBestModel();
        irisIter.reset();
        double score = bestNetwork.score(irisIter.next(), false);
        assertEquals(result.getBestModelScore(), score, 1e-2);
    }

    @Test
    public void testBadTuning() {
        //Test poor tuning (high LR): should terminate on MaxScoreIterationTerminationCondition

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(5.0)) //Intentionally huge LR
                        .weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3).activation(Activation.SOFTMAX)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5000))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES),
                                                        new MaxScoreIterationTerminationCondition(10)) //Initial score is ~2.5
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                                        .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, irisIter);
        EarlyStoppingResult result = trainer.fit();

        assertTrue(result.getTotalEpochs() < 5);
        assertEquals(EarlyStoppingResult.TerminationReason.IterationTerminationCondition,
                        result.getTerminationReason());
        String expDetails = new MaxScoreIterationTerminationCondition(10).toString();
        assertEquals(expDetails, result.getTerminationDetails());

        assertEquals(0, result.getBestModelEpoch());
        assertNotNull(result.getBestModel());
    }

    @Test
    public void testTimeTermination() {
        //test termination after max time

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(1e-6)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(10000))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(3, TimeUnit.SECONDS),
                                                        new MaxScoreIterationTerminationCondition(7.5)) //Initial score is ~2.5
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true))
                                        .modelSaver(saver).build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf, net, irisIter);
        long startTime = System.currentTimeMillis();
        EarlyStoppingResult result = trainer.fit();
        long endTime = System.currentTimeMillis();
        int durationSeconds = (int) (endTime - startTime) / 1000;

        assertTrue(durationSeconds >= 3);
        assertTrue(durationSeconds <= 9);

        assertEquals(EarlyStoppingResult.TerminationReason.IterationTerminationCondition,
                        result.getTerminationReason());
        String expDetails = new MaxTimeIterationTerminationCondition(3, TimeUnit.SECONDS).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }

    @Test
    public void testNoImprovementNEpochsTermination() {
        //Idea: terminate training if score (test set loss) does not improve for 5 consecutive epochs
        //Simulate this by setting LR = 0.0

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(0.0)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(100),
                                                        new ScoreImprovementEpochTerminationCondition(5))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(3, TimeUnit.SECONDS),
                                                        new MaxScoreIterationTerminationCondition(7.5)) //Initial score is ~2.5
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                                        .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, irisIter);
        EarlyStoppingResult result = trainer.fit();

        //Expect no score change due to 0 LR -> terminate after 6 total epochs
        assertEquals(6, result.getTotalEpochs());
        assertEquals(0, result.getBestModelEpoch());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition, result.getTerminationReason());
        String expDetails = new ScoreImprovementEpochTerminationCondition(5).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }

    @Test
    public void testMinImprovementNEpochsTermination() {
        //Idea: terminate training if score (test set loss) does not improve more than minImprovement for 5 consecutive epochs
        //Simulate this by setting LR = 0.0
        Random rng = new Random(123);
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Nesterovs(0.0,0.9)).list()
                        .layer(0, new DenseLayer.Builder().nIn(1).nOut(20)
                                        .weightInit(WeightInit.XAVIER).activation(
                                                        Activation.TANH)
                                        .build())
                        .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).weightInit(WeightInit.XAVIER)
                                        .activation(Activation.IDENTITY).weightInit(WeightInit.XAVIER).nIn(20).nOut(1)
                                        .build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));
        int nSamples = 100;
        //Generate the training data
        INDArray x = Nd4j.linspace(-10, 10, nSamples).reshape(nSamples, 1);
        INDArray y = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()));
        DataSet allData = new DataSet(x, y);

        List<DataSet> list = allData.asList();
        Collections.shuffle(list, rng);
        DataSetIterator training = new ListDataSetIterator(list, nSamples);

        double minImprovement = 0.0009;
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(1000),
                                                        //Go on for max 5 epochs without any improvements that are greater than minImprovement
                                                        new ScoreImprovementEpochTerminationCondition(5,
                                                                        minImprovement))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(3, TimeUnit.MINUTES))
                                        .scoreCalculator(new DataSetLossCalculator(training, true)).modelSaver(saver)
                                        .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, training);
        EarlyStoppingResult result = trainer.fit();

        assertEquals(6, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition, result.getTerminationReason());
        String expDetails = new ScoreImprovementEpochTerminationCondition(5, minImprovement).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }

    @Test
    public void testEarlyStoppingGetBestModel() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(0.001)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        MultipleEpochsIterator mIter = new MultipleEpochsIterator(10, irisIter);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                                        .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf, net, mIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        System.out.println(result);

        MultiLayerNetwork mln = result.getBestModel();

        assertEquals(net.getnLayers(), mln.getnLayers());
        assertEquals(net.conf().getOptimizationAlgo(), mln.conf().getOptimizationAlgo());
        BaseLayer bl = (BaseLayer) net.conf().getLayer();
        assertEquals(bl.getActivationFn().toString(), ((BaseLayer) mln.conf().getLayer()).getActivationFn().toString());
        assertEquals(bl.getIUpdater(), ((BaseLayer) mln.conf().getLayer()).getIUpdater());
    }

    @Test
    public void testListeners() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new Sgd(0.001)).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                        .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150, 150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                        new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                                        .iterationTerminationConditions(
                                                        new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                                        .build();

        LoggingEarlyStoppingListener listener = new LoggingEarlyStoppingListener();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, irisIter, listener);

        trainer.fit();

        assertEquals(1, listener.onStartCallCount);
        assertEquals(5, listener.onEpochCallCount);
        assertEquals(1, listener.onCompletionCallCount);
    }

    private static class LoggingEarlyStoppingListener implements EarlyStoppingListener<MultiLayerNetwork> {

        private static Logger log = LoggerFactory.getLogger(LoggingEarlyStoppingListener.class);
        private int onStartCallCount = 0;
        private int onEpochCallCount = 0;
        private int onCompletionCallCount = 0;

        @Override
        public void onStart(EarlyStoppingConfiguration esConfig, MultiLayerNetwork net) {
            log.info("EarlyStopping: onStart called");
            onStartCallCount++;
        }

        @Override
        public void onEpoch(int epochNum, double score, EarlyStoppingConfiguration esConfig, MultiLayerNetwork net) {
            log.info("EarlyStopping: onEpoch called (epochNum={}, score={}}", epochNum, score);
            onEpochCallCount++;
        }

        @Override
        public void onCompletion(EarlyStoppingResult esResult) {
            log.info("EarlyStopping: onCompletion called (result: {})", esResult);
            onCompletionCallCount++;
        }
    }


    @Test
    public void testRegressionScoreFunctionSimple() throws Exception {

        for(RegressionEvaluation.Metric metric : new RegressionEvaluation.Metric[]{RegressionEvaluation.Metric.MSE,
                RegressionEvaluation.Metric.MAE}) {
            log.info("Metric: " + metric);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new DenseLayer.Builder().nIn(784).nOut(32).build())
                    .layer(new OutputLayer.Builder().nIn(32).nOut(784).activation(Activation.SIGMOID).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSetIterator iter = new MnistDataSetIterator(32, false, 12345);

            List<DataSet> l = new ArrayList<>();
            for( int i=0; i<10; i++ ){
                DataSet ds = iter.next();
                l.add(new DataSet(ds.getFeatures(), ds.getFeatures()));
            }

            iter = new ExistingDataSetIterator(l);

            EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
            EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                            .iterationTerminationConditions(
                                    new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                            .scoreCalculator(new RegressionScoreCalculator(metric, iter)).modelSaver(saver)
                            .build();

            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iter);
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

            assertNotNull(result.getBestModel());
            assertTrue(result.getBestModelScore() > 0.0);
        }
    }

    @Test
    public void testAEScoreFunctionSimple() throws Exception {

        for(RegressionEvaluation.Metric metric : new RegressionEvaluation.Metric[]{RegressionEvaluation.Metric.MSE,
                RegressionEvaluation.Metric.MAE}) {
            log.info("Metric: " + metric);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new AutoEncoder.Builder().nIn(784).nOut(32).build())
                    .pretrain(true).backprop(false)
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSetIterator iter = new MnistDataSetIterator(32, false, 12345);

            List<DataSet> l = new ArrayList<>();
            for( int i=0; i<10; i++ ){
                DataSet ds = iter.next();
                l.add(new DataSet(ds.getFeatures(), ds.getFeatures()));
            }

            iter = new ExistingDataSetIterator(l);

            EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
            EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                            .iterationTerminationConditions(
                                    new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                            .scoreCalculator(new AutoencoderScoreCalculator(metric, iter)).modelSaver(saver)
                            .build();

            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iter);
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

            assertNotNull(result.getBestModel());
            assertTrue(result.getBestModelScore() > 0.0);
        }
    }

    @Test
    public void testVAEScoreFunctionSimple() throws Exception {

        for(RegressionEvaluation.Metric metric : new RegressionEvaluation.Metric[]{RegressionEvaluation.Metric.MSE,
                RegressionEvaluation.Metric.MAE}) {
            log.info("Metric: " + metric);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new VariationalAutoencoder.Builder()
                            .nIn(784).nOut(32)
                            .encoderLayerSizes(64)
                            .decoderLayerSizes(64)
                            .build())
                    .pretrain(true).backprop(false)
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSetIterator iter = new MnistDataSetIterator(32, false, 12345);

            List<DataSet> l = new ArrayList<>();
            for( int i=0; i<10; i++ ){
                DataSet ds = iter.next();
                l.add(new DataSet(ds.getFeatures(), ds.getFeatures()));
            }

            iter = new ExistingDataSetIterator(l);

            EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
            EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                            .iterationTerminationConditions(
                                    new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                            .scoreCalculator(new VAEReconErrorScoreCalculator(metric, iter)).modelSaver(saver)
                            .build();

            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iter);
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

            assertNotNull(result.getBestModel());
            assertTrue(result.getBestModelScore() > 0.0);
        }
    }

    @Test
    public void testVAEScoreFunctionReconstructionProbSimple() throws Exception {

        for(boolean logProb : new boolean[]{false, true}) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new VariationalAutoencoder.Builder()
                            .nIn(784).nOut(32)
                            .encoderLayerSizes(64)
                            .decoderLayerSizes(64)
                            .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                            .build())
                    .pretrain(true).backprop(false)
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSetIterator iter = new MnistDataSetIterator(32, false, 12345);

            List<DataSet> l = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                DataSet ds = iter.next();
                l.add(new DataSet(ds.getFeatures(), ds.getFeatures()));
            }

            iter = new ExistingDataSetIterator(l);

            EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
            EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                            .iterationTerminationConditions(
                                    new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                            .scoreCalculator(new VAEReconProbScoreCalculator(iter, 20, logProb)).modelSaver(saver)
                            .build();

            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iter);
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

            assertNotNull(result.getBestModel());
            assertTrue(result.getBestModelScore() > 0.0);
        }
    }

    @Test
    public void testClassificationScoreFunctionSimple() throws Exception {

        for(Evaluation.Metric metric : Evaluation.Metric.values()) {
            log.info("Metric: " + metric);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(new DenseLayer.Builder().nIn(784).nOut(32).build())
                    .layer(new OutputLayer.Builder().nIn(32).nOut(10).activation(Activation.SOFTMAX).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSetIterator iter = new MnistDataSetIterator(32, false, 12345);

            List<DataSet> l = new ArrayList<>();
            for( int i=0; i<10; i++ ){
                DataSet ds = iter.next();
                l.add(ds);
            }

            iter = new ExistingDataSetIterator(l);

            EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
            EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                    new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                            .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                            .iterationTerminationConditions(
                                    new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                            .scoreCalculator(new ClassificationScoreCalculator(metric, iter)).modelSaver(saver)
                            .build();

            EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf, net, iter);
            EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

            assertNotNull(result.getBestModel());
        }
    }

    @Test
    public void testEarlyStoppingListeners() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.001)).weightInit(WeightInit.XAVIER).list()
                .layer(0, new OutputLayer.Builder().nIn(4).nOut(3)
                        .lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);

        TestListener tl = new TestListener();
        net.setListeners(tl);

        DataSetIterator irisIter = new IrisDataSetIterator(50, 150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf =
                new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                        .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                        .iterationTerminationConditions(
                                new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                        .scoreCalculator(new DataSetLossCalculator(irisIter, true)).modelSaver(saver)
                        .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf, net, irisIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();

        assertEquals(5, tl.countEpochStart);
        assertEquals(5, tl.countEpochEnd);
        assertEquals(5 * 150/50, tl.iterCount);

        assertEquals(4, tl.maxEpochStart);
        assertEquals(4, tl.maxEpochEnd);
    }

    @Data
    public static class TestListener extends BaseTrainingListener {
        private int countEpochStart = 0;
        private int countEpochEnd = 0;
        private int iterCount = 0;
        private int maxEpochStart = -1;
        private int maxEpochEnd = -1;

        @Override
        public void onEpochStart(Model model){
            countEpochStart++;
            maxEpochStart = Math.max(maxEpochStart, BaseOptimizer.getEpochCount(model));
        }

        @Override
        public void onEpochEnd(Model model){
            countEpochEnd++;
            maxEpochEnd = Math.max(maxEpochEnd, BaseOptimizer.getEpochCount(model));
        }

        @Override
        public void iterationDone(Model model, int iteration, int epoch){
            iterCount++;
        }

    }
}
