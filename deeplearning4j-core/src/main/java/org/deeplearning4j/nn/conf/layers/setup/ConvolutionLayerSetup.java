package org.deeplearning4j.nn.conf.layers.setup;


import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;

import java.util.HashMap;
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
 *
 * @author Adam Gibson
 */
public class ConvolutionLayerSetup {

    private  int lastHeight = -1;
    private  int lastWidth = -1;
    private int lastnOut = -1;
    private  int lastOutChannels = -1;
    private int numLayers = -1;
    private Map<Integer,int[]> outSizesEachLayer = new HashMap<>();
    private Map<Integer,Integer> nInForLayer = new HashMap<>();
    /**
     * Take in the configuration
     * @param conf the configuration
     * @param height initial height of the data
     * @param width initial width of the data
     * @param channels initial number of channels in the data
     */
    public ConvolutionLayerSetup(MultiLayerConfiguration.Builder conf,int height,int width,int channels) {
        lastHeight = height;
        lastWidth = width;
        lastOutChannels = channels;

        if(conf instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder listBuilder = (NeuralNetConfiguration.ListBuilder) conf;
            numLayers = listBuilder.getLayerwise().size();
        }
        else
            numLayers = conf.getConfs().size();
        boolean alreadySet = false;

        for(int i = 0; i < numLayers; i++) {
            alreadySet = false;
            Layer curr = getLayer(i,conf);

            //cnn -> subsampling
            if(i == 0 || i < numLayers - 2 && curr instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer)getLayer(i,conf);
                //ensure the number of in channels is set for the data
                if(i == 0) {
                    convolutionLayer.setNIn(channels);
                    lastnOut = convolutionLayer.getNOut();
                }
                Layer next = getLayer(i + 1,conf);
                //cnn -> feedforward OR cnn -> rnn
                if(next instanceof DenseLayer || next instanceof OutputLayer || next instanceof BaseRecurrentLayer || next instanceof RnnOutputLayer ) {
                    //set the feed forward wrt the out channels of the current convolution layer
                    //set the rows and columns (height/width) wrt the kernel size of the current layer

                    int[] outWidthAndHeight = getConvolutionOutputSize(new int[]{lastHeight, lastWidth}, convolutionLayer.getKernelSize(), convolutionLayer.getPadding(), convolutionLayer.getStride());

                    if(next instanceof DenseLayer || next instanceof OutputLayer ) {
                        conf.inputPreProcessor(i + 1, new CnnToFeedForwardPreProcessor(
                                outWidthAndHeight[0]
                                , outWidthAndHeight[1], convolutionLayer.getNOut()));
                    } else {
                        conf.inputPreProcessor(i + 1, new CnnToRnnPreProcessor(
                                outWidthAndHeight[0]
                                , outWidthAndHeight[1], convolutionLayer.getNOut()));
                    }

                    //set the number of inputs wrt the current convolution layer
                    FeedForwardLayer o = (FeedForwardLayer) next;
                    outSizesEachLayer.put(i,outWidthAndHeight);
                    int outRows = outWidthAndHeight[0];
                    int outCols = outWidthAndHeight[1];
                    lastHeight = outRows;
                    lastWidth = outCols;
                    lastOutChannels = convolutionLayer.getNOut();
                    lastnOut = outCols * outRows * lastOutChannels;
                    nInForLayer.put(i, lastnOut);
                    o.setNIn(lastnOut);
                    alreadySet = true;
                }

                //cnn -> subsampling
                else if(next instanceof SubsamplingLayer) {
                    SubsamplingLayer subsamplingLayer = (SubsamplingLayer) next;
                    // subsamplingLayer.setKernelSize(convolutionLayer.getKernelSize());
                    if(subsamplingLayer.getPadding() == null)
                        subsamplingLayer.setPadding(convolutionLayer.getPadding());
                    lastnOut = convolutionLayer.getNOut();
                }
                //cnn -> cnn
                else if(next instanceof ConvolutionLayer) {
                    ConvolutionLayer nextConv = (ConvolutionLayer) next;
                    //set next layer's convolution input channels
                    //to be equal to this layer's out channels
                    lastOutChannels = lastnOut = convolutionLayer.getNOut();
                    nextConv.setNIn(lastOutChannels);
                    nInForLayer.put(i,lastOutChannels);
                }
            }

            else if(i < numLayers - 1 && curr instanceof SubsamplingLayer) {
                SubsamplingLayer subsamplingLayer = (SubsamplingLayer) getLayer(i,conf);
                Layer next = getLayer(i + 1,conf);

                //sub -> feedforward OR sub -> rnn
                if(next instanceof DenseLayer || next instanceof OutputLayer || next instanceof BaseRecurrentLayer || next instanceof RnnOutputLayer) {
                    //need to infer nins from first input size
                    int[] outWidthAndHeight = getSubSamplingOutputSize(new int[]{lastHeight, lastWidth}, subsamplingLayer.getKernelSize(), subsamplingLayer.getStride());
                    outSizesEachLayer.put(i,outWidthAndHeight);
                    int outRows = outWidthAndHeight[0];
                    int outCols = outWidthAndHeight[1];
                    lastHeight =  outWidthAndHeight[0];
                    lastWidth =  outWidthAndHeight[1];

                    //set the feed forward wrt the out channels of the current sub layer
                    //set the rows and columns (height/width) wrt the kernel size of the current layer
                    if( next instanceof DenseLayer || next instanceof OutputLayer ) {
                        conf.inputPreProcessor(i + 1, new CnnToFeedForwardPreProcessor(
                                outRows
                                , outCols, lastOutChannels));
                    } else if ( next instanceof RnnOutputLayer) {
                        conf.inputPreProcessor(i + 1, new CnnToRnnPreProcessor(
                                outRows
                                , outCols, lastOutChannels));
                    }
                    //set the number of inputs wrt the current sub layer
                    FeedForwardLayer o = (FeedForwardLayer) next;
                    lastnOut = outCols * outRows * lastOutChannels;
                    o.setNIn(lastnOut);
                    nInForLayer.put(i + 1, lastnOut);
                    //setup the fourd connections
                    setFourDtoTwoD(i, conf, o);
                    alreadySet = true;

                }

                //sub -> cnn
                else if(next instanceof ConvolutionLayer) {
                    ConvolutionLayer nextConv = (ConvolutionLayer) next;
                    //set next layer's convolution input channels
                    //to be equal to this layer's out channels
                    nextConv.setNIn(lastnOut);
                }
            }
            else if(i < numLayers - 1 && curr instanceof LocalResponseNormalization) {
                Layer next = getLayer(i + 1,conf);

                // LRN -> CNN
                if(next instanceof ConvolutionLayer){
                    ConvolutionLayer nextConv = (ConvolutionLayer) next;
                    //set next layer's convolution input channels
                    //to be equal to this layer's out channels
                    nextConv.setNIn(lastnOut);
                }
            }

            else if(i < numLayers - 1 && (curr instanceof DenseLayer || curr instanceof OutputLayer ||
                    getLayer(i,conf) instanceof BaseRecurrentLayer || getLayer(i,conf) instanceof RnnOutputLayer )) {
                FeedForwardLayer forwardLayer = (FeedForwardLayer) getLayer(i, conf);
                if(getLayer(i + 1,conf) instanceof  ConvolutionLayer) {
                    ConvolutionLayer convolutionLayer = (ConvolutionLayer) getLayer(i + 1,conf);
                    throw new UnsupportedOperationException("2d to 4d needs to be implemented");
                }
                else if(getLayer(i + 1,conf) instanceof SubsamplingLayer) {
                    SubsamplingLayer subsamplingLayer = (SubsamplingLayer) getLayer(i + 1,conf);
                    throw new UnsupportedOperationException("2d to 4d needs to be implemented");

                }
                //feedforward to feedforward
                else if(getLayer(i + 1,conf) instanceof OutputLayer || getLayer(i + 1,conf) instanceof DenseLayer) {
                    FeedForwardLayer d = (FeedForwardLayer) getLayer(i + 1,conf);
                    lastnOut = forwardLayer.getNOut();
                    d.setNIn(lastnOut);
                    nInForLayer.put(i + 1, lastnOut);
                }

                setFourDtoTwoD(i,conf,forwardLayer);
            }


            //cnn -> feed forward

            //feed forward to cnn
            //convolution to subsampling
            //subsampling to cnn
            //subsampling to feedforward
            //feedforward to subsampling
            //update the output size for a given activation
            //this allows us to track outputs for automatic setting
            //of certain values in the conv net
            if(curr instanceof ConvolutionLayer && i < numLayers - 1 && !alreadySet) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) curr;
                int[] outWidthAndHeight = getConvolutionOutputSize(new int[]{lastHeight, lastWidth}, convolutionLayer.getKernelSize(), convolutionLayer.getPadding(), convolutionLayer.getStride());
                lastHeight =  outWidthAndHeight[0];
                lastWidth =  outWidthAndHeight[1];
                lastOutChannels = convolutionLayer.getNOut();
                outSizesEachLayer.put(i,outWidthAndHeight);

            }
            //update the output size for a given
            //activation
            //this allows us to track outputs for automatic setting
            //of certain values in teh conv net
            else if(curr instanceof SubsamplingLayer && i < numLayers - 1 && !alreadySet) {
                SubsamplingLayer subsamplingLayer = (SubsamplingLayer) curr;
                int[] outWidthAndHeight = getSubSamplingOutputSize(new int[]{lastHeight, lastWidth}, subsamplingLayer.getKernelSize(), subsamplingLayer.getStride());
                lastHeight =  outWidthAndHeight[0];
                lastWidth =  outWidthAndHeight[1];
                outSizesEachLayer.put(i,outWidthAndHeight);

                //don't need channels here; its inferred from the last time
                //in the for loop
            }

        }

        if(getLayer(numLayers - 1,conf) instanceof OutputLayer || getLayer(numLayers - 1,conf) instanceof DenseLayer) {
            FeedForwardLayer lastLayer = (FeedForwardLayer) getLayer(numLayers - 1,conf);
            if(getLayer(numLayers - 2,conf) instanceof DenseLayer || getLayer(numLayers - 2,conf) instanceof OutputLayer) {
                FeedForwardLayer feedForwardLayer = (FeedForwardLayer) getLayer(numLayers - 2,conf);
                lastLayer.setNIn(feedForwardLayer.getNOut());
                lastnOut = feedForwardLayer.getNOut();
                nInForLayer.put(numLayers - 1, lastnOut);
            }
            else if(getLayer(numLayers - 2,conf) instanceof SubsamplingLayer) {
                lastnOut = lastHeight * lastWidth * lastOutChannels;
                lastLayer.setNIn(lastnOut);
                nInForLayer.put(numLayers - 1, lastnOut);
            }
            else if(getLayer(numLayers - 2,conf) instanceof ConvolutionLayer) {
                lastLayer.setNIn(lastnOut);
                nInForLayer.put(numLayers - 1, lastnOut);

            }
        }
        else if(getLayer(numLayers - 1,conf) instanceof ConvolutionLayer) {
            throw new UnsupportedOperationException("Unsupported path: final convolution layer");
        }
        else if(getLayer(numLayers - 1,conf) instanceof SubsamplingLayer) {
            throw new UnsupportedOperationException("Unsupported path: final subsampling layer");
        }

        if(conf instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder l = (NeuralNetConfiguration.ListBuilder) conf;
            if(l.getLayerwise().get(0).getLayer() instanceof ConvolutionLayer || l.getLayerwise().get(0).getLayer() instanceof SubsamplingLayer) {
                conf.inputPreProcessor(0,new FeedForwardToCnnPreProcessor(height,width,channels));
            }

        }
        else {
            if(conf.getConfs().get(0).getLayer() instanceof ConvolutionLayer || conf.getConfs().get(0).getLayer() instanceof SubsamplingLayer) {
                conf.inputPreProcessor(0,new FeedForwardToCnnPreProcessor(height,width,channels));
            }
        }




    }

    private int[] getSubSamplingOutputSize(int[] inputWidthAndHeight,int[] kernelWidthAndHeight,int[] stride) {
        int[] ret = new int[inputWidthAndHeight.length];
        for(int i = 0; i < ret.length; i++) {
            if(kernelWidthAndHeight[i] == 1)
                ret[i] = inputWidthAndHeight[i] / stride[i];
            else {
                ret[i] = (inputWidthAndHeight[i] - kernelWidthAndHeight[i]) / stride[i] + 1;
            }
        }

        return ret;
    }


    private int[] getConvolutionOutputSize(int[] inputWidthAndHeight, int[] kernelWidthAndHeight, int[] padding, int[] stride) {
        int[] ret = new int[inputWidthAndHeight.length];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (inputWidthAndHeight[i] - kernelWidthAndHeight[i] + (2 * padding[i])) / stride[i] + 1;
        }
        return ret;
    }

    public Layer getLayer(int i,MultiLayerConfiguration.Builder builder) {
        if(builder instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder listBuilder = (NeuralNetConfiguration.ListBuilder) builder;
            if(listBuilder.getLayerwise().get(i) == null)
                throw new IllegalStateException("Undefined layer " + i);
            return listBuilder.getLayerwise().get(i).getLayer();
        }

        return builder.getConfs().get(i).getLayer();
    }


    private void setFourDtoTwoD(int i, MultiLayerConfiguration.Builder conf, FeedForwardLayer d) {
        //only for output layer and dense layer
        if(d instanceof ConvolutionLayer)
            return;

        Layer currFourdLayer = conf instanceof NeuralNetConfiguration.ListBuilder  ? ((NeuralNetConfiguration.ListBuilder) conf).getLayerwise().get(i).getLayer() : conf.getConfs().get(i).getLayer();
        //2d -> 4d
        if(currFourdLayer instanceof ConvolutionLayer || currFourdLayer instanceof SubsamplingLayer) {
            if(currFourdLayer instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) currFourdLayer;
                int inputHeight = lastHeight;
                int inputWidth = lastWidth;
                //set the number of out such that the 2d output of
                //this layer match the width and height of the kernel
                /**
                 * We need to either set the number of in channels in the convolution layer
                 * to be equal to the outs of the dense/output layer
                 * or we need to set the number of outs
                 * equal to the kernel width and height.
                 *
                 * This allows the user flexibility in how they'd
                 * like to set the values.
                 *
                 */
                if(convolutionLayer.getKernelSize() != null) {
                    d.setNOut(inputHeight * inputWidth * convolutionLayer.getNOut());
                }
                else
                    throw new IllegalStateException("Unable to infer width and height without convolution layer kernel size");
                //set the input pre processor automatically for reshaping
                conf.inputPreProcessor(i + 1,new CnnToFeedForwardPreProcessor(inputHeight,inputWidth,lastOutChannels));

            }
            else if(currFourdLayer instanceof SubsamplingLayer) {
                int inputHeight = lastHeight;
                int inputWidth = lastWidth ;
                //set the number of out such that the 2d output of
                //this layer match the width and height of the kernel
                /**
                 * We need to either set the number of in channels in the convolution layer
                 * to be equal to the outs of the dense/output layer
                 * or we need to set the number of outs
                 * equal to the kernel width and height.
                 *
                 * This allows the user flexibility in how they'd
                 * like to set the values.
                 *
                 *
                 * Note here that we only modify the output layer's
                 * nouts when the next fully connected layer
                 * isn't the final one.
                 */

                //set the input pre processor automatically for reshaping
                conf.inputPreProcessor(i + 1,new CnnToFeedForwardPreProcessor(inputHeight,inputWidth,lastOutChannels));


            }

        }
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

    public Map<Integer, int[]> getOutSizesEachLayer() {
        return outSizesEachLayer;
    }

    public void setOutSizesEachLayer(Map<Integer, int[]> outSizesEachLayer) {
        this.outSizesEachLayer = outSizesEachLayer;
    }

    public Map<Integer, Integer> getnInForLayer() {
        return nInForLayer;
    }

    public void setnInForLayer(Map<Integer, Integer> nInForLayer) {
        this.nInForLayer = nInForLayer;
    }
}
