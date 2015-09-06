package org.deeplearning4j.nn.layers.convolution;


import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;

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
    private  int lastOutChannels = -1;

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

        int numLayers = -1;
        if(conf instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder listBuilder = (NeuralNetConfiguration.ListBuilder) conf;
            numLayers = listBuilder.getLayerwise().size();
        }
        else
            numLayers = conf.getConfs().size();
        for(int i = 0; i < numLayers; i++) {
            Layer curr = getLayer(i,conf);
            //cnn -> subsampling
            if(i < numLayers - 1 && getLayer(i, conf) instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer)getLayer(i,conf);
                Layer next = getLayer(i + 1,conf);
                //cnn -> feedforward
                if(next instanceof FeedForwardLayer) {
                    //set the feed forward wrt the out channels of the current convolution layer
                    //set the rows and columns (height/width) wrt the kernel size of the current layer
                    conf.inputPreProcessor(i,new CnnToFeedForwardPreProcessor(
                            convolutionLayer.getKernelSize()[0]
                            ,convolutionLayer.getKernelSize()[1],convolutionLayer.getNOut()));
                    //set the number of inputs wrt the current convolution layer
                    FeedForwardLayer o = (FeedForwardLayer) next;
                    //need to infer nins from first input size
                    int outRows = lastHeight - convolutionLayer.getKernelSize()[0] + convolutionLayer.getPadding()[0];
                    int outCols = lastWidth - convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
                    int nIn = outCols * outRows * convolutionLayer.getNOut();
                    o.setNIn(nIn);


                }
                //cnn -> subsampling
                else if(next instanceof SubsamplingLayer) {
                    SubsamplingLayer subsamplingLayer = (SubsamplingLayer) next;
                    subsamplingLayer.setKernelSize(convolutionLayer.getKernelSize());
                    if(subsamplingLayer.getPadding() == null)
                        subsamplingLayer.setPadding(convolutionLayer.getPadding());
                }
                //cnn -> cnn
                else if(next instanceof ConvolutionLayer) {
                    ConvolutionLayer nextConv = (ConvolutionLayer) next;
                    //set next layer's convolution input channels
                    //to be equal to this layer's out channels
                    nextConv.setNIn(convolutionLayer.getNOut());
                }
            }

            else if(i < numLayers - 1 && getLayer(i,conf) instanceof SubsamplingLayer) {
                SubsamplingLayer subsamplingLayer = (SubsamplingLayer) getLayer(i,conf);
                Layer next = getLayer(i + 1,conf);
                //cnn -> feedforward
                if(next instanceof FeedForwardLayer) {
                    //set the feed forward wrt the out channels of the current convolution layer
                    //set the rows and columns (height/width) wrt the kernel size of the current layer
                    conf.inputPreProcessor(i,new CnnToFeedForwardPreProcessor(
                            subsamplingLayer.getKernelSize()[0]
                            ,subsamplingLayer.getKernelSize()[1],lastOutChannels));
                    //set the number of inputs wrt the current convolution layer
                    FeedForwardLayer o = (FeedForwardLayer) next;
                    //need to infer nins from first input size
                    int outRows = lastHeight - subsamplingLayer.getKernelSize()[0] + subsamplingLayer.getPadding()[0];
                    int outCols = lastWidth - subsamplingLayer.getKernelSize()[1] + subsamplingLayer.getPadding()[1];
                    int nIn = outCols * outRows * lastOutChannels;
                    o.setNIn(nIn);


                }
                //cnn -> subsampling

                //cnn -> cnn
                else if(next instanceof ConvolutionLayer) {
                    ConvolutionLayer nextConv = (ConvolutionLayer) next;
                    //set next layer's convolution input channels
                    //to be equal to this layer's out channels
                    nextConv.setNIn(lastOutChannels);
                }
            }


            else if(i < conf.getConfs().size() - 1 && conf.getConfs().get(i).getLayer() instanceof FeedForwardLayer) {
                FeedForwardLayer d = (FeedForwardLayer) conf.getConfs().get(i).getLayer();
                setFourDtoTwoD(i,conf,d);
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
            if(curr instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) curr;
                lastHeight -=  convolutionLayer.getKernelSize()[0] + convolutionLayer.getPadding()[0];
                lastWidth -=  convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
                lastOutChannels = convolutionLayer.getNOut();

            }
            //update the output size for a given
            //activation
            //this allows us to track outputs for automatic setting
            //of certain values in teh conv net
            else if(curr instanceof SubsamplingLayer) {
                SubsamplingLayer subsamplingLayer = (SubsamplingLayer) curr;
                lastHeight /= subsamplingLayer.getStride()[0];
                lastWidth /= subsamplingLayer.getStride()[1];
                //don't need channels here; its inferred from the last time
                //in the for loop
            }

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

    public Layer getLayer(int i,MultiLayerConfiguration.Builder builder) {
        if(builder instanceof NeuralNetConfiguration.ListBuilder) {
            NeuralNetConfiguration.ListBuilder listBuilder = (NeuralNetConfiguration.ListBuilder) builder;
            return listBuilder.getLayerwise().get(i).getLayer();
        }

        return builder.getConfs().get(i).getLayer();
    }


    private void setFourDtoTwoD(int i, MultiLayerConfiguration.Builder conf, FeedForwardLayer d) {
        Layer next = conf instanceof NeuralNetConfiguration.ListBuilder  ? ((NeuralNetConfiguration.ListBuilder) conf).getLayerwise().get(i + 1).getLayer() : conf.getConfs().get(i).getLayer();
        //2d -> 4d
        if(next instanceof ConvolutionLayer || next instanceof SubsamplingLayer) {
            if(next instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) next;
                int inputHeight = lastHeight - convolutionLayer.getKernelSize()[0]  + convolutionLayer.getPadding()[0];
                int inputWidth = lastWidth - convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
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
                if(convolutionLayer.getKernelSize() != null){
                    d.setNOut(inputHeight * inputWidth * convolutionLayer.getNOut());
                }
                else
                    throw new IllegalStateException("Unable to infer width and height without convolution layer kernel size");
                //set the input pre processor automatically for reshaping
                conf.inputPreProcessor(i + 1,new CnnToFeedForwardPreProcessor(inputHeight,inputWidth,lastOutChannels));

            }
            else if(next instanceof SubsamplingLayer) {
                SubsamplingLayer convolutionLayer = (SubsamplingLayer) next;
                int inputHeight = lastHeight - convolutionLayer.getKernelSize()[0]  + convolutionLayer.getPadding()[0];
                int inputWidth = lastWidth - convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
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
                if(convolutionLayer.getKernelSize() != null){
                    d.setNOut(inputHeight * inputWidth * lastOutChannels);
                }
                else
                    throw new IllegalStateException("Unable to infer width and height without convolution layer kernel size");

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
}
