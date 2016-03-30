package org.deeplearning4j.earlystopping;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.listener.EarlyStoppingListener;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxScoreIterationTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

public class TestEarlyStopping {

    @Test
    public void testEarlyStoppingIris(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0,new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(irisIter,true))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf,net,irisIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        System.out.println(result);

        assertEquals(5, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition,result.getTerminationReason());
        Map<Integer,Double> scoreVsIter = result.getScoreVsEpoch();
        assertEquals(5,scoreVsIter.size());
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
    public void testEarlyStoppingIrisMultiEpoch(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0,new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);
        MultipleEpochsIterator mIter = new MultipleEpochsIterator(10, irisIter);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(irisIter,true))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf,net,mIter);

        EarlyStoppingResult<MultiLayerNetwork> result = trainer.fit();
        System.out.println(result);

        assertEquals(5, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition,result.getTerminationReason());
        Map<Integer,Double> scoreVsIter = result.getScoreVsEpoch();
        assertEquals(5,scoreVsIter.size());
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
    public void testBadTuning(){
        //Test poor tuning (high LR): should terminate on MaxScoreIterationTerminationCondition

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD).learningRate(1.0)    //Intentionally huge LR
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5000))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES),
                        new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                .scoreCalculator(new DataSetLossCalculator(irisIter, true))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,irisIter);
        EarlyStoppingResult result = trainer.fit();

        assertTrue(result.getTotalEpochs() < 5);
        assertEquals(EarlyStoppingResult.TerminationReason.IterationTerminationCondition, result.getTerminationReason());
        String expDetails = new MaxScoreIterationTerminationCondition(7.5).toString();
        assertEquals(expDetails, result.getTerminationDetails());

        assertEquals(0, result.getBestModelEpoch());
        assertNotNull(result.getBestModel());
    }

    @Test
    public void testTimeTermination(){
        //test termination after max time

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD).learningRate(1e-6)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0,new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(10000))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(3, TimeUnit.SECONDS),
                        new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                //.scoreCalculator(new DataSetLossCalculator(irisIter, true))   //No score calculator in this test (don't need score)
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new EarlyStoppingTrainer(esConf,net,irisIter);
        long startTime = System.currentTimeMillis();
        EarlyStoppingResult result = trainer.fit();
        long endTime = System.currentTimeMillis();
        int durationSeconds = (int)(endTime-startTime)/1000;

        assertTrue(durationSeconds >= 3);
        assertTrue(durationSeconds <= 9);

        assertEquals(EarlyStoppingResult.TerminationReason.IterationTerminationCondition, result.getTerminationReason());
        String expDetails = new MaxTimeIterationTerminationCondition(3,TimeUnit.SECONDS).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }

    @Test
    public void testNoImprovementNEpochsTermination(){
        //Idea: terminate training if score (test set loss) does not improve for 5 consecutive epochs
        //Simulate this by setting LR = 0.0

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD).learningRate(0.0)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0,new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100),
                        new ScoreImprovementEpochTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(3, TimeUnit.SECONDS),
                        new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                .scoreCalculator(new DataSetLossCalculator(irisIter, true))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,irisIter);
        EarlyStoppingResult result = trainer.fit();

        //Expect no score change due to 0 LR -> terminate after 6 total epochs
        assertEquals(6, result.getTotalEpochs());
        assertEquals(0, result.getBestModelEpoch());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition,result.getTerminationReason());
        String expDetails = new ScoreImprovementEpochTerminationCondition(5).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }

    @Test
    public void testMinImprovementNEpochsTermination(){
        //Idea: terminate training if score (test set loss) does not improve more than minImprovement for 5 consecutive epochs
        //Simulate this by setting LR = 0.0
        Random rng = new Random(123);
        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .iterations(10)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(0.001)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(1).nOut(20)
                        .weightInit(WeightInit.XAVIER)
                        .activation("tanh")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .weightInit(WeightInit.XAVIER)
                        .activation("identity").weightInit(WeightInit.XAVIER)
                        .nIn(20).nOut(1).build())
                .pretrain(false).backprop(true).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));
        int nSamples = 100;
        //Generate the training data
        INDArray x = Nd4j.linspace(-10,10,nSamples).reshape(nSamples, 1);
        INDArray y = Nd4j.getExecutioner().execAndReturn(new Sin(x.dup()));
        DataSet allData = new DataSet(x,y);

        List<DataSet> list = allData.asList();
        Collections.shuffle(list,rng);
        DataSetIterator training = new ListDataSetIterator(list,nSamples);

        double minImprovement = 0.0009;
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(1000),
                        //Go on for max 5 epochs without any improvements that are greater than minImprovement
                        new ScoreImprovementEpochTerminationCondition(5, minImprovement))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(3, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(training, true))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,training);
        EarlyStoppingResult result = trainer.fit();

        //The test ends at 28 epochs (Nothing is random, so it will always end the same)
        assertEquals(28, result.getTotalEpochs());
        //The last epoch (27) has the best data so far
        assertEquals(27, result.getBestModelEpoch());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition,result.getTerminationReason());
        String expDetails = new ScoreImprovementEpochTerminationCondition(5, minImprovement).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }


    @Test
    public void testListeners(){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0,new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(irisIter,true))
                .modelSaver(saver)
                .build();

        LoggingEarlyStoppingListener listener = new LoggingEarlyStoppingListener();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,irisIter,listener);

        trainer.fit();

        assertEquals(1,listener.onStartCallCount);
        assertEquals(5,listener.onEpochCallCount);
        assertEquals(1,listener.onCompletionCallCount);
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
            log.info("EarlyStopping: onEpoch called (epochNum={}, score={}}",epochNum,score);
            onEpochCallCount++;
        }

        @Override
        public void onCompletion(EarlyStoppingResult esResult) {
            log.info("EarlyStopping: onCompletion called (result: {})",esResult);
            onCompletionCallCount++;
        }
    }
}
