package org.deeplearning4j.nn.conf.weightnoise;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.rng.distribution.factory.DefaultDistributionFactory;
import org.nd4j.linalg.api.rng.distribution.factory.DistributionFactory;
import org.nd4j.linalg.api.rng.distribution.impl.NormalDistribution;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.SigmoidSchedule;

public class TestWeightNoise {

    @Test
    public void testWeightNoiseConfigJson(){

        IWeightNoise[] weightNoises = new IWeightNoise[]{
                new DropConnect(0.5),
                new DropConnect(new SigmoidSchedule(ScheduleType.ITERATION, 0.5, 0.5, 100)),
                new WeightNoise(new NormalDistribution(0, 0.1))
        };

        for(IWeightNoise wn : weightNoises) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightNoise(wn)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                    .layer(new DenseLayer.Builder().nIn(10).nOut(10).weightNoise(new DropConnect(0.25)).build())
                    .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();;
        }



    }

}
