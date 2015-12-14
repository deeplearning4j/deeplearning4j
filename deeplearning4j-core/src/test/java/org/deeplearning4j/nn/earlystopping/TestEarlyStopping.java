package org.deeplearning4j.nn.earlystopping;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.earlystopping.trainer.IEarlyStoppingTrainer;
import org.deeplearning4j.nn.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.nn.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.nn.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class TestEarlyStopping {

    @Test
    public void testEarlyStoppingIris(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.SGD)
                .weightInit(WeightInit.XAVIER)
                .list(1)
                .layer(0,new OutputLayer.Builder().nIn(4).nOut(3).lossFunction(LossFunctions.LossFunction.MCXENT).build())
                .pretrain(false).backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(new ScoreIterationListener(1));

        DataSetIterator irisIter = new IrisDataSetIterator(150,150);
        EarlyStoppingModelSaver saver = new InMemoryModelSaver();
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(new MaxEpochsTerminationCondition(5))
                .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(1, TimeUnit.MINUTES))
                .scoreCalculator(new DataSetLossCalculator(irisIter,true))
                .modelSaver(saver)
                .build();

        IEarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,net,irisIter);
        trainer.fit();

        EarlyStoppingResult result = trainer.fit();
        System.out.println(result);

        assertEquals(5, result.getTotalEpochs());
        assertEquals(EarlyStoppingResult.TerminationReason.EpochTerminationCondition,result.getTerminationReason());
        Map<Integer,Double> scoreVsIter = result.getScoreVsEpoch();
        assertEquals(5,scoreVsIter.size());
        String expDetails = esConf.getEpochTerminationConditions().get(0).toString();
        assertEquals(expDetails, result.getTerminationDetails());

        MultiLayerNetwork out = result.getBestModel();
        assertNotNull(out);
    }
}
