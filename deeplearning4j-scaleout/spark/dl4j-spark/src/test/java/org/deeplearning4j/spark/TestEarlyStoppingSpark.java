package org.deeplearning4j.spark;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
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
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.earlystopping.SparkDataSetLossCalculator;
import org.deeplearning4j.spark.earlystopping.SparkEarlyStoppingTrainer;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class TestEarlyStoppingSpark extends BaseSparkTest {

    @Test
    public void testEarlyStoppingIris() {
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


        JavaRDD<DataSet> irisData = getIris();

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                .scoreCalculator(new SparkDataSetLossCalculator(irisData,true,sc.sc()))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new SparkEarlyStoppingTrainer(getContext().sc(),esConf,net,irisData,50,150,5);

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
        double score = bestNetwork.score(new IrisDataSetIterator(150,150).next());
        assertEquals(result.getBestModelScore(), score, 1e-3);
    }

    @Test
    public void testBadTuning() {
        //Test poor tuning (high LR): should terminate on MaxScoreIterationTerminationCondition

        Nd4j.getRandom().setSeed(12345);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD).learningRate(2.0)    //Intentionally huge LR
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        JavaRDD<DataSet> irisData = getIris();
        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5000))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES),
                        new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                .scoreCalculator(new SparkDataSetLossCalculator(irisData,true,sc.sc()))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new SparkEarlyStoppingTrainer(getContext().sc(),esConf,net,irisData,50,150,5);
        EarlyStoppingResult result = trainer.fit();

        assertTrue(result.getTotalEpochs() < 5);
        assertEquals(EarlyStoppingResult.TerminationReason.IterationTerminationCondition, result.getTerminationReason());
        String expDetails = new MaxScoreIterationTerminationCondition(7.5).toString();
        assertEquals(expDetails, result.getTerminationDetails());
    }

    @Test
    public void testTimeTermination() {
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

        JavaRDD<DataSet> irisData = getIris();

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(10000))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(3, TimeUnit.SECONDS),
                        new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                .scoreCalculator(new SparkDataSetLossCalculator(irisData,true,sc.sc()))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new SparkEarlyStoppingTrainer(getContext().sc(),esConf,net,irisData,50,150,5);
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
    public void testNoImprovementNEpochsTermination() {
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

        JavaRDD<DataSet> irisData = getIris();

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(100),
                        new ScoreImprovementEpochTerminationCondition(5))
                .iterationTerminationConditions(new MaxScoreIterationTerminationCondition(7.5))  //Initial score is ~2.5
                .scoreCalculator(new SparkDataSetLossCalculator(irisData,true,sc.sc()))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new SparkEarlyStoppingTrainer(getContext().sc(),esConf,net,irisData,50,150,5);
        EarlyStoppingResult result = trainer.fit();

        //Expect no score change due to 0 LR -> terminate after 6 total epochs
        assertTrue(result.getTotalEpochs()<12);  //Normally expect 6 epochs exactly; get a little more than that here due to rounding + order of operations
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition,result.getTerminationReason());
        String expDetails = new ScoreImprovementEpochTerminationCondition(5).toString();
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


        JavaRDD<DataSet> irisData = getIris();

        EarlyStoppingModelSaver<MultiLayerNetwork> saver = new InMemoryModelSaver<>();
        EarlyStoppingConfiguration<MultiLayerNetwork> esConf = new EarlyStoppingConfiguration.Builder<MultiLayerNetwork>()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                .scoreCalculator(new SparkDataSetLossCalculator(irisData,true,sc.sc()))
                .modelSaver(saver)
                .build();

        LoggingEarlyStoppingListener listener = new LoggingEarlyStoppingListener();

        IEarlyStoppingTrainer<MultiLayerNetwork> trainer = new SparkEarlyStoppingTrainer(getContext().sc(),esConf,net,irisData,50,150,5,listener);

        trainer.fit();

        assertEquals(1,listener.onStartCallCount);
        assertEquals(5,listener.onEpochCallCount);
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
            log.info("EarlyStopping: onEpoch called (epochNum={}, score={}}",epochNum,score);
            onEpochCallCount++;
        }

        @Override
        public void onCompletion(EarlyStoppingResult esResult) {
            log.info("EorlyStopping: onCompletion called (result: {})",esResult);
            onCompletionCallCount++;
        }
    }

    private JavaRDD<DataSet> getIris(){

        JavaSparkContext sc = getContext();

        IrisDataSetIterator iter = new IrisDataSetIterator(1,150);
        List<DataSet> list = new ArrayList<>(150);
        while(iter.hasNext()) list.add(iter.next());

        return sc.parallelize(list);
    }
}
