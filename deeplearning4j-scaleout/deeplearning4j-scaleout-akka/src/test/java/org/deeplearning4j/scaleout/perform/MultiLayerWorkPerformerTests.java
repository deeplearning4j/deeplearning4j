package org.deeplearning4j.scaleout.perform;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.IrisDataFetcher;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;
import org.junit.Test;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class MultiLayerWorkPerformerTests extends NeuralNetWorkPerformerTest {

    @Test
    public void testDbn() {
        RandomGenerator gen = new MersenneTwister(123);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().withActivationType(NeuralNetConfiguration.ActivationType.NET_ACTIVATION)
                .momentum(9e-1f).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.normal(gen, 1e-1))
                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen).iterations(10)
                .learningRate(1e-1f).nIn(4).nOut(3).list(2).hiddenLayerSizes(new int[]{3}) .override(new NeuralNetConfiguration.ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {

                        if (i == 1) {
                            builder.weightInit(WeightInit.ZERO);
                            builder.activationFunction(Activations.softMaxRows());
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);

                        }
                    }
                }).build();

        String json = conf.toJson();

        Configuration conf2 = new Configuration();
        conf2.set(DeepLearningConfigurable.MULTI_LAYER_CONF,json);
        conf2.set(DeepLearningConfigurable.CLASS,DBN.class.getName());
        WorkerPerformer performer = new BaseMultiLayerNetworkWorkPerformer();
        performer.setup(conf2);
        IrisDataFetcher fetcher = new IrisDataFetcher();
        fetcher.fetch(10);
        DataSet d = fetcher.next();
        Job j = new Job(d,"1");
        assumeJobResultNotNull(performer,j);
        performer.update(j.getResult());

    }


}
