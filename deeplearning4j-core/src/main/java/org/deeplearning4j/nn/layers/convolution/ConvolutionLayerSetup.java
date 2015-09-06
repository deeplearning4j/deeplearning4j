package org.deeplearning4j.nn.layers.convolution;

import javassist.bytecode.analysis.SubroutineScanner;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.nd4j.linalg.util.ArrayUtil;

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
    /**
     * Take in the configuration
     * @param conf the configuration
     * @param height initial height of the data
     * @param width initial width of the data
     * @param channels initial number of channels in the data
     */
    public ConvolutionLayerSetup(NeuralNetConfiguration.ListBuilder conf,int height,int width,int channels) {
        int lastHeight = height;
        int lastWidth = width;
        int lastOutChannels = channels;
        for(int i = 0; i < conf.getLayerwise().size(); i++) {
            Layer curr =  conf.getLayerwise().get(i).getLayer();
            //cnn -> subsampling
            if(i < conf.getLayerwise().size() - 1 && conf.getLayerwise().get(i).getLayer() instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) conf.getLayerwise().get(i).getLayer();
                Layer next = conf.getLayerwise().get(i + 1).getLayer();
                //cnn -> feedforward
                if(next instanceof DenseLayer || next instanceof OutputLayer) {
                    //set the feed forward wrt the out channels of the current convolution layer
                    //set the rows and columns (height/width) wrt the kernel size of the current layer
                    conf.inputPreProcessor(i,new CnnToFeedForwardPreProcessor(
                            convolutionLayer.getKernelSize()[0]
                            ,convolutionLayer.getKernelSize()[1],convolutionLayer.getNOut()));
                    //set the number of inputs wrt the current convolution layer
                    if(next instanceof DenseLayer) {
                        DenseLayer d = (DenseLayer) next;
                        //need to infer nins from first input size
                        int outRows = lastHeight - convolutionLayer.getKernelSize()[0] + convolutionLayer.getPadding()[0];
                        int outCols = lastWidth - convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
                        int nIn = outCols * outRows * convolutionLayer.getNOut();
                        d.setNIn(nIn);
                    }
                    if(next instanceof OutputLayer) {
                        OutputLayer o = (OutputLayer) next;
                        //need to infer nins from first input size
                        int outRows = lastHeight - convolutionLayer.getKernelSize()[0] + convolutionLayer.getPadding()[0];
                        int outCols = lastWidth - convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
                        int nIn = outCols * outRows * convolutionLayer.getNOut();
                        o.setNIn(nIn);
                    }

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

            else if(i < conf.getLayerwise().size() - 1 && conf.getLayerwise().get(i).getLayer() instanceof OutputLayer) {
                OutputLayer o = (OutputLayer) conf.getLayerwise().get(i).getLayer();
                Layer next = conf.getLayerwise().get(i + 1).getLayer();
                //2d -> 4d
                if(next instanceof ConvolutionLayer || next instanceof SubsamplingLayer) {
                    ConvolutionLayer convolutionLayer = (ConvolutionLayer) next;

                }

            }

            else if(i < conf.getLayerwise().size() - 1 && conf.getLayerwise().get(i).getLayer() instanceof DenseLayer) {
                DenseLayer d = (DenseLayer) conf.getLayerwise().get(i).getLayer();
                Layer next = conf.getLayerwise().get(i + 1).getLayer();
                //2d -> 4d
                if(next instanceof ConvolutionLayer || next instanceof SubsamplingLayer) {
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
                    if(d.getNOut() > 0) {

                    }
                    else if(convolutionLayer.getKernelSize() != null){

                    }
                    d.setNOut(inputHeight * inputWidth * convolutionLayer.getNOut());

                }
            }


            //cnn -> feed forward

            //feed forward to cnn
            //convolution to subsampling
            //subsampling to cnn
            //subsampling to feedforward
            //feedforward to subsampling
            if(curr instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) curr;
                lastHeight -=  convolutionLayer.getKernelSize()[0] + convolutionLayer.getPadding()[0];
                lastWidth -=  convolutionLayer.getKernelSize()[1] + convolutionLayer.getPadding()[1];
                lastOutChannels = convolutionLayer.getNOut();

            }
            else if(curr instanceof SubsamplingLayer) {
                SubsamplingLayer subsamplingLayer = (SubsamplingLayer) curr;
                lastHeight /= subsamplingLayer.getStride()[0];
                lastWidth /= subsamplingLayer.getStride()[1];
                //don't need channels here; its inferred from the last time
                //in the for loop
            }

        }

        if(conf.getLayerwise().get(0).getLayer() instanceof ConvolutionLayer || conf.getLayerwise().get(0).getLayer() instanceof SubsamplingLayer) {
            conf.inputPreProcessor(0,new FeedForwardToCnnPreProcessor(height,width,channels));
        }

    }

}
