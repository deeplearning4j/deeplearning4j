package org.deeplearning4j;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class Debug {

    @Test
    public void debug6() {
        Nd4j.getRandom().setSeed(12345);

        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        int depthIn = 2;
        int depthOut = 2;
        int nOut = 2;
        int width = 3;

        PoolingType pt = PoolingType.SUM;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same).seed(12345L).list()
                .layer(0, new ConvolutionLayer.Builder().nIn(depthIn).nOut(depthOut).kernelSize(2, width)
                        .hasBias(false)
                        .stride(1, width).activation(Activation.TANH).build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer.Builder().poolingType(pt)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(depthOut).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf.clone());
        net.init();
        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);
        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.SEPARATE);

//        //No workspace: passes
//        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.NONE);
//        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.NONE);

//        //Single workspace: passes
//        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.SINGLE);
//        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.SINGLE);

        INDArray in = Nd4j.rand(new int[]{1, 2, 5, 3});
        INDArray out1 = net.output(in);
        INDArray out2 = net.output(in);

        System.out.println(Arrays.toString(out1.dup().data().asFloat()));
        System.out.println(Arrays.toString(out2.dup().data().asFloat()));

        assertEquals(out1, out2);

    }

    @Test
    public void debug7() {
        Nd4j.getRandom().setSeed(12345);

        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
        int depthIn = 2;
        int depthOut = 2;
        int nOut = 2;
        int width = 3;

        PoolingType pt = PoolingType.SUM;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same).seed(12345L).list()
//                .layer(0, new ConvolutionLayer.Builder().nIn(depthIn).nOut(depthOut).kernelSize(2, width)
//                        .stride(1, width).activation(Activation.TANH).build())
                .layer(0, new org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer.Builder().poolingType(pt)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(depthOut).nOut(nOut).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf.clone());
        net.init();
        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);
        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.SEPARATE);

//        //No workspace: passes
//        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.NONE);
//        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.NONE);

//        //Single workspace: passes
//        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.SINGLE);
//        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.SINGLE);

        INDArray in = Nd4j.rand(new int[]{1, 2, 5, 1});
        INDArray out1 = net.output(in);
        INDArray out2 = net.output(in);

        System.out.println(Arrays.toString(out1.dup().data().asFloat()));
        System.out.println(Arrays.toString(out2.dup().data().asFloat()));

        assertEquals(out1, out2);

    }

}
