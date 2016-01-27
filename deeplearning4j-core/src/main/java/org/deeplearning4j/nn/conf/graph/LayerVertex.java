package org.deeplearning4j.nn.conf.graph;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.inputs.InvalidInputTypeException;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.factory.LayerFactories;

import java.util.Arrays;

/**
 * LayerVertex is a GraphVertex with a Layer configuration (and, optionally preprocessor) in it
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class LayerVertex extends GraphVertex {

    private NeuralNetConfiguration layerConf;
    private InputPreProcessor preProcessor;

    @Override
    public GraphVertex clone() {
        return new LayerVertex(layerConf.clone(), (preProcessor != null ? preProcessor.clone() : null));
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof LayerVertex)) return false;
        LayerVertex lv = (LayerVertex) o;
        if (!layerConf.equals(lv.layerConf)) return false;
        if (preProcessor == null && lv.preProcessor != null || preProcessor != null && lv.preProcessor == null)
            return false;
        return preProcessor == null || preProcessor.equals(lv.preProcessor);
    }

    @Override
    public int hashCode() {
        return layerConf.hashCode() ^ (preProcessor != null ? preProcessor.hashCode() : 0);
    }

    @Override
    public org.deeplearning4j.nn.graph.vertex.GraphVertex instantiate(ComputationGraph graph, String name, int idx) {
        return new org.deeplearning4j.nn.graph.vertex.impl.LayerVertex(
                graph, name, idx,
                LayerFactories.getFactory(layerConf).create(layerConf, null, idx),
                preProcessor);
    }

    @Override
    public InputType getOutputType(InputType... vertexInputs) throws InvalidInputTypeException {
        if (vertexInputs.length != 1) {
            throw new InvalidInputTypeException("LayerVertex expects exactly one input. Got: " + Arrays.toString(vertexInputs));
        }

        //Assume any necessary preprocessors have already been added
        Layer layer = layerConf.getLayer();
        if (layer instanceof ConvolutionLayer || layer instanceof SubsamplingLayer) {
            InputType.InputTypeConvolutional afterPreProcessor;
            if (preProcessor != null) {
                if (preProcessor instanceof FeedForwardToCnnPreProcessor) {
                    FeedForwardToCnnPreProcessor ffcnn = (FeedForwardToCnnPreProcessor) preProcessor;
                    afterPreProcessor = (InputType.InputTypeConvolutional) InputType.convolutional(ffcnn.getNumChannels(), ffcnn.getInputWidth(), ffcnn.getInputHeight());
                } else if (preProcessor instanceof RnnToCnnPreProcessor) {
                    RnnToCnnPreProcessor rnncnn = (RnnToCnnPreProcessor) preProcessor;
                    afterPreProcessor = (InputType.InputTypeConvolutional) InputType.convolutional(rnncnn.getNumChannels(), rnncnn.getInputWidth(), rnncnn.getInputHeight());
                } else {
                    //Assume no change to type of input...
                    //TODO checks for non convolutional input...
                    afterPreProcessor = (InputType.InputTypeConvolutional) vertexInputs[0];
                }
            } else {
                afterPreProcessor = (InputType.InputTypeConvolutional) vertexInputs[0];
            }

            int channelsOut;
            int[] kernel;
            int[] stride;
            int[] padding;
            if (layer instanceof ConvolutionLayer) {
                channelsOut = ((ConvolutionLayer) layer).getNOut();
                kernel = ((ConvolutionLayer) layer).getKernelSize();
                stride = ((ConvolutionLayer) layer).getStride();
                padding = ((ConvolutionLayer) layer).getPadding();
            } else {
                channelsOut = afterPreProcessor.getDepth();
                kernel = ((SubsamplingLayer) layer).getKernelSize();
                stride = ((SubsamplingLayer) layer).getStride();
                padding = ((SubsamplingLayer) layer).getPadding();
            }

            //First: check that the kernel size/stride/padding is valid
            int inWidth = afterPreProcessor.getWidth();
            int inHeight = afterPreProcessor.getHeight();
            //Check filter > size + padding
            if (kernel[0] > inWidth + padding[0] || kernel[1] > inHeight + padding[1]) {
                throw new InvalidInputTypeException("Invalid input: activations into layer are w=" + inWidth + ", h=" + inHeight
                        + " but kernel size is " + Arrays.toString(kernel) + " with padding " + Arrays.toString(padding));
            }

            //Check proposed filter/padding size actually works:
            if ((inWidth - kernel[0] + 2 * padding[0]) % stride[0] != 0) {
                throw new InvalidInputTypeException("Invalid input/configuration: activations into layer are inputWidth=" + inWidth + ", widthPadding=" + padding[0]
                        + ", kernelWidth = " + kernel[0] + ", strideWidth = " + stride[0] + ". (inputWidth-kernelWidth+2*widthPadding)/strideWidth is not an integer");
            }
            if ((inHeight - kernel[1] + 2 * padding[1]) % stride[1] != 0) {
                throw new InvalidInputTypeException("Invalid input/configuration: activations into layer are inputHeight=" + inHeight + ", heightPadding=" + padding[1]
                        + ", kernelHeight = " + kernel[1] + ", strideHeight = " + stride[1] + ". (inputHeight-kernelHeight+2*heightPadding)/strideHeight is not an integer");
            }

            int outWidth = (inWidth - kernel[0] + 2 * padding[0]) / stride[0] + 1;
            int outHeight = (inHeight - kernel[1] + 2 * padding[1]) / stride[1] + 1;

            return InputType.convolutional(channelsOut,outWidth,outHeight);
        } else if (layer instanceof BaseRecurrentLayer) {
            return InputType.recurrent();
        } else if (layer instanceof FeedForwardLayer) {
            //Dense, autoencoder, etc
            return InputType.feedForward();
        } else {
            //Unknown... probably same as input??
            return vertexInputs[0];
        }
    }
}
