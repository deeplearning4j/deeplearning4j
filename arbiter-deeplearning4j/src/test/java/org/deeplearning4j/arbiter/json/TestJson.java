package org.deeplearning4j.arbiter.json;

import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.layers.ConvolutionLayerSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.junit.Test;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 14/02/2017.
 */
public class TestJson {

    @Test
    public void testMultiLayerSpaceJson(){

        MultiLayerSpace mls = new MultiLayerSpace.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(new ContinuousParameterSpace(0.0001, 0.2))
                .regularization(true)
                .l2(new ContinuousParameterSpace(0.0001, 0.05))
                .dropOut(new ContinuousParameterSpace(0.2, 0.7))
                .iterations(1)
                .addLayer(new ConvolutionLayerSpace.Builder()
                        .nIn(1).nOut(new IntegerParameterSpace(5, 30))
                        .kernelSize(new DiscreteParameterSpace<>(new int[]{3, 3}, new int[]{4, 4}, new int[]{5, 5}))
                        .stride(new DiscreteParameterSpace<>(new int[]{1, 1}, new int[]{2, 2}))
                        .activation(new DiscreteParameterSpace<>("relu","softplus","leakyrelu"))
                        .build(), new IntegerParameterSpace(1, 2), true) //1-2 identical layers
                .addLayer(new DenseLayerSpace.Builder().nIn(4).nOut(new IntegerParameterSpace(2, 10))
                        .activation(new DiscreteParameterSpace<String>("relu", "tanh"))
                        .build(), new IntegerParameterSpace(0, 1), true)   //0 to 1 layers
                .addLayer(new OutputLayerSpace.Builder().nOut(10).activation("softmax")
                        .iLossFunction(LossFunctions.LossFunction.MCXENT.getILossFunction()).build())
                .setInputType(InputType.convolutional(28,28,1))
                .pretrain(false).backprop(true).build();

        String asJson = mls.toJson();
        String asYaml = mls.toYaml();

        System.out.println(asJson);
        System.out.println(asYaml);

        MultiLayerSpace fromJson = MultiLayerSpace.fromJson(asJson);
        MultiLayerSpace fromYaml = MultiLayerSpace.fromYaml(asYaml);

        assertEquals(mls, fromJson);
        assertEquals(mls, fromYaml);
    }

    @Test
    public void testComputationGraphSpaceJson(){

        fail();
    }

    @Test
    public void testOptimizationConfigurationJson(){

        fail();
    }

}
