package org.deeplearning4j.nn.misc;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@Slf4j
public class WorkspaceTests {

    @Before
    public void before(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @After
    public void after(){
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
    }

    @Test
    public void checkScopesTestCGAS() throws Exception {
        ComputationGraph c = createNet();
        for(WorkspaceMode wm : new WorkspaceMode[]{WorkspaceMode.SEPARATE, WorkspaceMode.SINGLE}) {
            log.info("Starting test: {}", wm);
            c.getConfiguration().setTrainingWorkspaceMode(wm);
            c.getConfiguration().setInferenceWorkspaceMode(wm);

            INDArray f = Nd4j.rand(new int[]{8, 1, 28, 28});
            INDArray l = Nd4j.rand(8, 10);
            c.setInputs(f);
            c.setLabels(l);

            c.computeGradientAndScore();
        }
    }


    @Test
    public void testWorkspaceIndependence() {
        //https://github.com/deeplearning4j/deeplearning4j/issues/4337
        int depthIn = 2;
        int depthOut = 2;
        int nOut = 2;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same).seed(12345L).list()
                .layer(0, new ConvolutionLayer.Builder().nIn(depthIn).nOut(depthOut).kernelSize(2, 2)
                        .stride(1, 1).activation(Activation.TANH).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(nOut).build())
                .setInputType(InputType.convolutional(5,5,2))
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf.clone());
        net.init();
        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.SEPARATE);
        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.SEPARATE);

        MultiLayerNetwork net2 = new MultiLayerNetwork(conf.clone());
        net2.init();
        net2.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.NONE);
        net2.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.NONE);

        INDArray in = Nd4j.rand(new int[]{1,2,5,5});

        net.output(in);
        net2.output(in);    //Op [add_scalar] X argument uses leaked workspace pointer from workspace [LOOP_EXTERNAL]
    }

    public static ComputationGraph createNet() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new ConvolutionLayer.Builder().nOut(3)
                        .kernelSize(2,2).stride(2,2).build(), "in")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(3)
                        .kernelSize(2,2).stride(2,2).build(), "0")
                .addLayer("out", new OutputLayer.Builder().nOut(10)
                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "1")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(28,28,1))
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

}
