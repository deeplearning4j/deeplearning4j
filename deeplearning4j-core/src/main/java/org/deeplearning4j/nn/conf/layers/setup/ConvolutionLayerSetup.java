package org.deeplearning4j.nn.conf.layers.setup;


import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * Automatic configuration of convolutional layers:
 * Handles all layer wise interactions
 * between convolution/subsampling -> dense/output
 * convolution -> subsampling
 *
 * among others.
 *
 * It does this by tracking a moving window
 * of all the various configurations through
 * out the network.
 *
 * The moving window tracks everything from the
 * out channels of the previous layer
 * as well as the different interactions
 * such as when a shift from
 * convolution to dense happens.
 *
 */
public class ConvolutionLayerSetup {

    protected int lastHeight = -1;
    protected int lastWidth = -1;
    protected int lastOutChannels = -1;
    protected int lastnOut = -1;
    protected int numLayers = -1;
    protected String inLayerName;
    protected String outLayerName;
    protected Map<String,int[]> nOutsPerLayer = new HashMap<>();
    protected Map<String,Integer> nInsPerLayer = new HashMap<>();
    protected MultiLayerConfiguration.Builder conf;

    /**
     * Take in the configuration
     * @param builder the configuration builder
     * @param height initial height of the data
     * @param width initial width of the data
     * @param channels initial number of channels in the data
     */

    public ConvolutionLayerSetup(MultiLayerConfiguration.Builder builder,int height,int width,int channels) {
        conf = builder;
        lastHeight = height;
        lastWidth = width;
        lastOutChannels = channels;

        if(conf instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder listBuilder = (NeuralNetConfiguration.ListBuilder) conf;
            numLayers = listBuilder.getLayerwise().size();
        } else {
            numLayers = conf.getConfs().size();
        }
        for(int i = 0; i < numLayers-1; i++) {
            Layer inputLayer = getLayer(i,conf);
            Layer outputLayer = getLayer(i+1,conf);
            updateLayerInputs(i, inputLayer, outputLayer);
        }
    }

    private void storeNInAndNOut(String inName, int out){
        nInsPerLayer.put(inName, out);
        nOutsPerLayer.put(inLayerName, new int[]{lastHeight, lastWidth, lastOutChannels});
    }

    private void updateLayerInputs(int i, Layer inputLayer, Layer outputLayer){
        int lastLayerNumber = numLayers - 1;
        inLayerName = (inputLayer.getLayerName() != null) ? inputLayer.getLayerName(): Integer.toString(i);
        outLayerName = (outputLayer.getLayerName() != null) ? outputLayer.getLayerName(): Integer.toString(i+1);

        if(i < lastLayerNumber){
            switch (inputLayer.getClass().getSimpleName()){
                case "ConvolutionLayer":
                    ConvolutionLayer convolutionLayer = (ConvolutionLayer) inputLayer;
                    if(i == 0) {
                        conf.inputPreProcessor(i, new FeedForwardToCnnPreProcessor(lastHeight, lastWidth, lastOutChannels));
                        lastnOut = convolutionLayer.getNOut();
                        convolutionLayer.setNIn(lastOutChannels);
                    }
                    getConvolutionOutputSize(new int[]{lastHeight, lastWidth}, convolutionLayer.getKernelSize(), convolutionLayer.getPadding(), convolutionLayer.getStride());
                    lastOutChannels = convolutionLayer.getNOut();
                    switch (outputLayer.getClass().getSimpleName()) {
                        case "ConvolutionLayer":
                            ConvolutionLayer nextConv = (ConvolutionLayer) outputLayer;
                            //set next layer's convolution input channels to be equal to this layer's out channels
                            lastOutChannels = lastnOut = convolutionLayer.getNOut();
                            storeNInAndNOut(inLayerName, lastnOut);
                            nextConv.setNIn(lastnOut);
                            break;
                        case "SubsamplingLayer":
                            SubsamplingLayer subsamplingLayer = (SubsamplingLayer) outputLayer;
                            lastOutChannels = lastnOut = convolutionLayer.getNOut();
                            storeNInAndNOut(inLayerName, lastnOut);
                            break;
                        //cnn -> feedforward OR cnn -> rnn
                        case "DenseLayer":
                        case "OutputLayer":
                            FeedForwardLayer feedForwardLayer = (FeedForwardLayer) outputLayer;
                            lastnOut = lastHeight * lastWidth * lastOutChannels;
                            storeNInAndNOut(inLayerName, lastnOut); // required to be before inputPreProcessor to update lastHeight and lastWidth
                            feedForwardLayer.setNIn(lastnOut);
                            conf.inputPreProcessor(i + 1, new CnnToFeedForwardPreProcessor(lastHeight, lastWidth, lastOutChannels));
                            break;
                        case "BaseRecurrentLayer":
                        case "RnnOutputLayer":
                            feedForwardLayer = (FeedForwardLayer) outputLayer;
                            lastnOut = lastHeight * lastWidth * lastOutChannels;
                            storeNInAndNOut(inLayerName, lastnOut);
                            feedForwardLayer.setNIn(lastnOut);
                            conf.inputPreProcessor(i + 1, new CnnToRnnPreProcessor(lastHeight, lastWidth, lastOutChannels));
                            break;
                        case "LocalResponseNormalization":
                            LocalResponseNormalization lrn = (LocalResponseNormalization) outputLayer;
                            lastOutChannels = lastnOut = convolutionLayer.getNOut();
                            storeNInAndNOut(inLayerName, lastOutChannels);
                    }
                    break;
                case "SubsamplingLayer":
                    if(i < lastLayerNumber){
                        SubsamplingLayer subsamplingLayer = (SubsamplingLayer) inputLayer;
                        getSubSamplingOutputSize(new int[]{lastHeight, lastWidth}, subsamplingLayer.getKernelSize(), subsamplingLayer.getStride());
                        if (i == 0) throw new UnsupportedOperationException("Unsupported path: first layer shouldn't be " + inputLayer.getClass().getSimpleName());
                        switch (outputLayer.getClass().getSimpleName()) {
                            // ccn -> ccn
                            case "ConvolutionLayer":
                                ConvolutionLayer nextConv = (ConvolutionLayer) outputLayer;
                                storeNInAndNOut(outLayerName, lastOutChannels);
                                nextConv.setNIn(lastOutChannels);
                                break;
                            case "SubsamplingLayer":
                                //no op
                                break;
                            //sub -> ffn
                            case "DenseLayer":
                            case "OutputLayer":
                                FeedForwardLayer feedForwardLayer = (FeedForwardLayer) outputLayer;
                                lastnOut = lastHeight * lastWidth * lastOutChannels;
                                storeNInAndNOut(outLayerName, lastnOut);
                                feedForwardLayer.setNIn(lastnOut);
                                conf.inputPreProcessor(i + 1, new CnnToFeedForwardPreProcessor(lastHeight, lastWidth, lastOutChannels));
                                break;
                            // sub -> rnn
                            case "BaseRecurrentLayer":
                            case "RnnOutputLayer":
                                feedForwardLayer = (FeedForwardLayer) outputLayer;
                                lastnOut = lastHeight * lastWidth * lastOutChannels;
                                storeNInAndNOut(outLayerName, lastnOut);
                                feedForwardLayer.setNIn(lastnOut);
                                conf.inputPreProcessor(i + 1, new CnnToRnnPreProcessor(lastHeight, lastWidth, lastOutChannels));
                                break;
                        }
                    }
                    break;
                case "DenseLayer":
                case "BaseRecurrentLayer":
                case "RnnOutputLayer":
                    if (i == 0) throw new UnsupportedOperationException("Apply nIn attribute to the layer configuration for " + inputLayer.getClass().getSimpleName());
                    FeedForwardLayer feedForwardLayer = (FeedForwardLayer) inputLayer;
                    switch (outputLayer.getClass().getSimpleName()) {
                        // ffn -> ccn
                        case "ConvolutionLayer":
                            convolutionLayer = (ConvolutionLayer) outputLayer;
                            conf.inputPreProcessor(i, new FeedForwardToCnnPreProcessor(lastHeight, lastWidth, lastOutChannels));
                            lastnOut = convolutionLayer.getNOut();
                            convolutionLayer.setNIn(lastnOut);
                            // ffn -> sub
                        case "SubsamplingLayer":
                            throw new UnsupportedOperationException("Subsampling Layer should be connected to Convolution Layer");
                        case "DenseLayer":
                        case "OutputLayer":
                        case "BaseRecurrentLayer":
                        case "RnnOutputLayer":
                            FeedForwardLayer feedForwardLayer2 = (FeedForwardLayer) outputLayer;
                            lastnOut = feedForwardLayer.getNOut();
                            storeNInAndNOut(outLayerName, lastnOut);
                            feedForwardLayer2.setNIn(lastnOut);
                            break;
                    }
                    break;
                case "OutputLayer":
                    throw new UnsupportedOperationException("OutputLayer should be the last layer");
                case "LocalResponseNormalization":
                    if (i == 0) throw new UnsupportedOperationException("Unsupported path: first layer shouldn't be " + inputLayer.getClass().getSimpleName());
                    switch (outputLayer.getClass().getSimpleName()) {
                        //sub -> feedforward || rnn
                        case "DenseLayer":
                        case "OutputLayer":
                        case "BaseRecurrentLayer":
                        case "RnnOutputLayer":
                            feedForwardLayer = (FeedForwardLayer) outputLayer;
                            feedForwardLayer.setNIn(lastnOut);
                            break;
                        // lrn -> ccn
                        case "ConvolutionLayer":
                            ConvolutionLayer nextConv = (ConvolutionLayer) outputLayer;
                            nextConv.setNIn(lastnOut);
                    }
                    break;
            }
        } else
            throw new UnsupportedOperationException("Unsupported path: final " + inputLayer.getClass().getSimpleName() + " layer");
    }


    private void getSubSamplingOutputSize(int[] inputWidthAndHeight,int[] kernelWidthAndHeight,int[] stride) {
        int[] ret = new int[inputWidthAndHeight.length];
        for(int i = 0; i < ret.length; i++) {
            if(kernelWidthAndHeight[i] == 1)
                ret[i] = inputWidthAndHeight[i] / stride[i];
            else {
                ret[i] = (inputWidthAndHeight[i] - kernelWidthAndHeight[i]) / stride[i] + 1;
            }
        }
        lastHeight = ret[0];
        lastWidth = ret[1];
    }


    private void getConvolutionOutputSize(int[] inputWidthAndHeight, int[] kernelWidthAndHeight, int[] padding, int[] stride) {
        int[] ret = new int[inputWidthAndHeight.length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (inputWidthAndHeight[i] - kernelWidthAndHeight[i] + (2 * padding[i])) / stride[i] + 1;
        }
        lastHeight = ret[0];
        lastWidth = ret[1];
    }

    public Layer getLayer(int i, MultiLayerConfiguration.Builder builder) {
        if(builder instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder listBuilder = (NeuralNetConfiguration.ListBuilder) builder;
            if(listBuilder.getLayerwise().get(i) == null)
                throw new IllegalStateException("Undefined layer " + i);
            return listBuilder.getLayerwise().get(i).getLayer();
        }

        return builder.getConfs().get(i).getLayer();
    }

    public int getLastHeight() {
        return lastHeight;
    }

    public void setLastHeight(int lastHeight) {
        this.lastHeight = lastHeight;
    }

    public int getLastWidth() {
        return lastWidth;
    }

    public void setLastWidth(int lastWidth) {
        this.lastWidth = lastWidth;
    }

    public int getLastOutChannels() {
        return lastOutChannels;
    }

    public void setLastOutChannels(int lastOutChannels) {
        this.lastOutChannels = lastOutChannels;
    }

    public Map<String, int[]> getOutSizesEachLayer() {
        return nOutsPerLayer;
    }

    public void setOutSizesEachLayer(Map<String, int[]> outSizesEachLayer) {
        this.nOutsPerLayer = outSizesEachLayer;
    }

    public Map<String, Integer> getnInForLayer() {
        return nInsPerLayer;
    }

    public void setnInForLayer(Map<String, Integer> nInForLayer) {
        this.nInsPerLayer = nInForLayer;
    }
}