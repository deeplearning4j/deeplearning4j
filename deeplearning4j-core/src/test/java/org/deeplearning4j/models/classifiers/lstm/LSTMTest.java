package org.deeplearning4j.models.classifiers.lstm;

import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 12/30/14.
 */
public class LSTMTest {

    private static Logger log = LoggerFactory.getLogger(LSTMTest.class);

    @Test
    public void testTraffic() {
        List<String> vocab = Arrays.asList("Skymind", "does", "deep", "learning");
        LayerFactory factory = LayerFactories.getFactory(LSTM.class);

        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
                .layerFactory(factory)
                .nIn(4).nOut(4).build();
        LSTM l = factory.create(conf);
        INDArray predict = FeatureUtil.toOutcomeMatrix(new int[]{0,1,2,3},4);
        l.fit(predict);
    }

}
