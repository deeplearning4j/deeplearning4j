package org.deeplearning4j.nn.conf.constraints;

import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.constraint.MaxNormConstraint;
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint;
import org.deeplearning4j.nn.conf.constraint.NonNegativeConstraint;
import org.deeplearning4j.nn.conf.constraint.UnitNormConstraint;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestConstraints {

    @Test
    public void testConstraints(){


        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for(LayerConstraint lc : constraints){

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .constraints(lc)
                    .learningRate(0.0)
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,5))
                    .list()
                    .layer(new DenseLayer.Builder().nIn(12).nOut(10).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            List<LayerConstraint> exp = Collections.singletonList(lc.clone());
            assertEquals(exp, net.getLayer(0).conf().getLayer().getConstraints());
            assertEquals(exp, net.getLayer(1).conf().getLayer().getConstraints());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray w0 = net.getParam("0_W");
            INDArray b0 = net.getParam("0_b");
            INDArray w1 = net.getParam("1_W");
            INDArray b1 = net.getParam("1_b");

            if(lc instanceof MaxNormConstraint){
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.5 );
                assertTrue(w1.norm2(1).maxNumber().doubleValue() <= 0.5 );
            } else if(lc instanceof MinMaxNormConstraint){
                assertTrue(w0.norm2(1).minNumber().doubleValue() >= 0.3 );
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.4 );
                assertTrue(w1.norm2(1).minNumber().doubleValue() >= 0.3 );
                assertTrue(w1.norm2(1).maxNumber().doubleValue() <= 0.4 );
            } else if(lc instanceof NonNegativeConstraint ){
                assertTrue(w0.minNumber().doubleValue() >= 0.0 );
            } else if(lc instanceof UnitNormConstraint ){
                assertEquals(w0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6 );
                assertEquals(w0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6 );
                assertEquals(w1.norm2(1).minNumber().doubleValue(), 1.0, 1e-6 );
                assertEquals(w1.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6 );
            }
        }

    }

}
