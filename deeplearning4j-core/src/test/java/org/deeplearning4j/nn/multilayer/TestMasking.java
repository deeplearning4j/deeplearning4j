package org.deeplearning4j.nn.multilayer;

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;

import static org.junit.Assert.assertNull;

/**
 * Created by Alex on 20/01/2017.
 */
public class TestMasking {

    @Test
    public void checkMaskArrayClearance(){
        for(boolean tbptt : new boolean[]{true, false}) {
            //Simple "does it throw an exception" type test...
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .iterations(1).seed(12345).list()
                    .layer(0, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .activation("identity").nIn(1).nOut(1).build())
                    .backpropType(tbptt ? BackpropType.TruncatedBPTT : BackpropType.Standard).tBPTTForwardLength(8).tBPTTBackwardLength(8)
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            DataSet data = new DataSet(Nd4j.linspace(1, 10, 10).reshape(1, 1, 10), Nd4j.linspace(2, 20, 10).reshape(1, 1, 10),
                    Nd4j.ones(10), Nd4j.ones(10));

            net.fit(data);
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }


            net.fit(data.getFeatures(), data.getLabels(), data.getFeaturesMaskArray(), data.getLabelsMaskArray());
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }

            DataSetIterator iter = new ExistingDataSetIterator(Collections.singletonList(data).iterator());
            net.fit(iter);
            for (Layer l : net.getLayers()) {
                assertNull(l.getMaskArray());
            }
        }
    }

}
