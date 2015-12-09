package org.arbiter.deeplearning4j;

import org.apache.commons.lang.ArrayUtils;
import org.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.arbiter.optimize.parameter.FixedValue;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestLayerSpace {

    @Test
    public void testBasic(){

        DenseLayer expected = new DenseLayer.Builder()
                .nIn(10).nOut(20).build();

        LayerSpace ls = new LayerSpace.Builder()
                .layer(DenseLayer.class)
                .add("nIn", new FixedValue<Integer>(10))
                .add("nOut", new FixedValue<Integer>(20))
                .build();

        List<Layer> layerList = ls.randomLayers();
        assertEquals(1, layerList.size());

        Layer l = layerList.get(0);
        assertTrue(l instanceof DenseLayer);

        DenseLayer actual = (DenseLayer)l;

        assertEquals(expected,actual);
    }

    @Test
    public void testBasic2(){

        String[] actFns = new String[]{"softsign","relu","leakyrelu"};

        for( int i=0; i<20; i++ ) {
            LayerSpace ls = new LayerSpace.Builder()
                    .layer(DenseLayer.class)
                    .add("nIn", new FixedValue<Integer>(10))
                    .add("nOut", new FixedValue<Integer>(20))
                    .add("learningRate", new ContinuousParameterSpace(0.3, 0.4))
                    .add("l2", new ContinuousParameterSpace(0.01,0.1))
                    .add("activation", new DiscreteParameterSpace<String>(actFns))
                    .build();

            List<Layer> layerList = ls.randomLayers();
            assertEquals(1, layerList.size());

            DenseLayer l = (DenseLayer) layerList.get(0);
            assertEquals(10, l.getNIn());
            assertEquals(20, l.getNOut());
            double lr = l.getLearningRate();
            double l2 = l.getL2();
            String activation = l.getActivationFunction();

            System.out.println(lr + "\t" + l2 + "\t" + activation);

            assertTrue(lr >= 0.3 && lr <= 0.4);
            assertTrue(l2 >= 0.01 && l2 <= 0.1 );
            assertTrue(ArrayUtils.contains(actFns,activation));
        }
    }

}
