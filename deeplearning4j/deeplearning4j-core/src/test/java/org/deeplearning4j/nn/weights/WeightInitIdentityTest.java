package org.deeplearning4j.nn.weights;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.junit.Test;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link WeightInitIdentity}
 *
 * @author Christian Skarby
 */
public class WeightInitIdentityTest {

    /**
     * Test identity mapping for 1d convolution
     */
    @Test
    public void testIdConv1D() {
        final INDArray input = Nd4j.randn(new long[] {1,5,7});
        final String inputName = "input";
        final String conv = "conv";
        final String output = "output";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.inferInputType(input))
                .addInputs(inputName)
                .setOutputs(output)
                .layer(conv, new Convolution1DLayer.Builder(7)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(input.size(1))
                        .weightInit(new WeightInitIdentity())
                        .activation(new ActivationIdentity())
                        .build(), inputName)
                .layer(output, new RnnLossLayer.Builder().activation(new ActivationIdentity()).build(), conv)
                .build());
        graph.init();

        assertEquals("Mapping was not identity!", input, graph.outputSingle(input).reshape(input.shape()));
    }

    /**
     * Test identity mapping for 2d convolution
     */
    @Test
    public void testIdConv2D() {
        final INDArray input = Nd4j.randn(1,5,7,11);
        final String inputName = "input";
        final String conv = "conv";
        final String output = "output";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.inferInputType(input))
                .addInputs(inputName)
                .setOutputs(output)
                .layer(conv, new ConvolutionLayer.Builder(3,5)
                        .convolutionMode(ConvolutionMode.Same)
                        .nOut(input.size(1))
                        .weightInit(new WeightInitIdentity())
                        .activation(new ActivationIdentity())
                        .build(), inputName)
                .layer(output, new CnnLossLayer.Builder().activation(new ActivationIdentity()).build(), conv)
                .build());
        graph.init();

        assertEquals("Mapping was not identity!", input, graph.outputSingle(input));
    }

    /**
     * Test identity mapping for 3d convolution
     */
    @Test
    public void testIdConv3D() {
        final INDArray input = Nd4j.randn(1,5,7,11,13);
        final String inputName = "input";
        final String conv = "conv";
        final String output = "output";
        final ComputationGraph graph = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .setInputTypes(InputType.inferInputType(input))
                .addInputs(inputName)
                .setOutputs(output)
                .layer(conv, new Convolution3D.Builder(3,7,5)
                        .convolutionMode(ConvolutionMode.Same)
                        .dataFormat(Convolution3D.DataFormat.NCDHW)
                        .nOut(input.size(1))
                        .weightInit(new WeightInitIdentity())
                        .activation(new ActivationIdentity())
                        .build(), inputName)
                .layer(output, new Cnn3DLossLayer.Builder(Convolution3D.DataFormat.NCDHW).activation(new ActivationIdentity()).build(), conv)
                .build());
        graph.init();

        assertEquals("Mapping was not identity!", input, graph.outputSingle(input));
    }
}
