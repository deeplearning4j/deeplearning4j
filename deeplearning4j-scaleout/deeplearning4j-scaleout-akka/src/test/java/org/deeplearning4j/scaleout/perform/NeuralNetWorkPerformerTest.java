package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.junit.Test;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class NeuralNetWorkPerformerTest extends BaseWorkPerformerTest {
    @Test
    public void testRbm() {
        NeuralNetConfiguration conf = new NeuralNetConfiguration
                .Builder().nIn(4).nOut(3).build();
        String json = conf.toJson();
        Configuration conf2 = new Configuration();
        conf2.set(DeepLearningConfigurable.NEURAL_NET_CONF,json);
        conf2.set(DeepLearningConfigurable.CLASS,RBM.class.getName());
        WorkerPerformer performer = new NeuralNetWorkPerformer();
        performer.setup(conf2);
    }


}
