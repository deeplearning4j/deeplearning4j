/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */



package org.deeplearning4j.nn.conf;

import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
public class MultiLayerConfiguration implements Serializable, Cloneable {

    protected List<NeuralNetConfiguration> confs;
    protected boolean pretrain = true;
    protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected boolean backprop = false;
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;
    //whether to redistribute params or not
    @Deprecated
    protected boolean redistributeParams = false;

    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toYaml() {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        try {
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromYaml(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }



    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toJson() {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromJson(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return toJson();
    }



    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }


    @Override
    public MultiLayerConfiguration clone() {
        try {
            MultiLayerConfiguration clone = (MultiLayerConfiguration) super.clone();

            if(clone.confs != null) {
                List<NeuralNetConfiguration> list = new ArrayList<>();
                for(NeuralNetConfiguration conf : clone.confs) {
                    list.add(conf.clone());
                }
                clone.confs = list;
            }

            if(clone.inputPreProcessors != null) {
                Map<Integer,InputPreProcessor> map = new HashMap<>();
                for(Map.Entry<Integer,InputPreProcessor> entry : clone.inputPreProcessors.entrySet()) {
                    map.put(entry.getKey(), entry.getValue().clone());
                }
                clone.inputPreProcessors = map;
            }

            return clone;

        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public InputPreProcessor getInputPreProcess(int curr) {
        return inputPreProcessors.get(curr);
    }

    @Data
    public static class Builder {

        protected List<NeuralNetConfiguration> confs = new ArrayList<>();
        protected boolean pretrain = true;
        protected double dampingFactor = 100;
        protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
        protected boolean backprop = false;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = 20;
        protected int tbpttBackLength = 20;
        @Deprecated
        protected boolean redistributeParams = false;
        protected int[] cnnInputSize = null;    //Order: height/width/depth

        /**
         * Whether to redistribute parameters as a view or not
         * @param redistributeParams whether to redistribute parameters
         *                           as a view or not
         * @return
         */
        @Deprecated
        public Builder redistributeParams(boolean redistributeParams) {
            this.redistributeParams = redistributeParams;
            return this;
        }

        /**
         * Specify the processors.
         * These are used at each layer for doing things like normalization and
         * shaping of input.
         * @param processor what to use to preProcess the data.
         * @return builder pattern
         */
        public Builder inputPreProcessor(Integer layer, InputPreProcessor processor) {
            inputPreProcessors.put(layer,processor);
            return this;
        }

        public Builder inputPreProcessors(Map<Integer,InputPreProcessor> processors) {
            this.inputPreProcessors = processors;
            return this;
        }

        /**
         * Whether to do back prop or not
         * @param backprop whether to do back prop or not
         * @return
         */
        public Builder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }
        
        /**The type of backprop. Default setting is used for most networks (MLP, CNN etc),
         * but optionally truncated BPTT can be used for training recurrent neural networks.
         * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
         */
        public Builder backpropType(BackpropType type){
        	this.backpropType = type;
        	return this;
        }
        
        /**When doing truncated BPTT: how many steps of forward pass should we do
         * before doing (truncated) backprop?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
         * but may be larger than it in some circumstances (but never smaller)<br>
         * Ideally your training data time series length should be divisible by this
         * This is the k1 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         * @param forwardLength Forward length > 0, >= backwardLength
         */
        public Builder tBPTTForwardLength(int forwardLength){
        	this.tbpttFwdLength = forwardLength;
        	return this;
        }
        
        /**When doing truncated BPTT: how many steps of backward should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * This is the k2 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         * @param backwardLength <= forwardLength
         */
        public Builder tBPTTBackwardLength(int backwardLength){
        	this.tbpttBackLength = backwardLength;
        	return this;
        }

        /**
         * Whether to do pre train or not
         * @param pretrain whether to do pre train or not
         * @return builder pattern
         */
        public Builder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;
        }

        /** Size of input for CNNs. Should only be used when convolutional layers are present.
         * This information (input size) is necessary in order to:<br>
         * (a) automatically add layer preprocessors, which allow CNN and dense/output layers to be used together
         * (as well as CNN/RNN layers) <br>
         * (b) automatically calculate input/output sizes; for example, input size for a dense layer, in a
         *     Convolutional->DenseLayer or Convolutional->OutputLayer configuratio
         * @param height Input image height
         * @param width Input image width
         * @param depth Input image depth / number of channels (for example: 3 for color, 1 for grayscale etc)
         */
        public Builder cnnInputSize(int height, int width, int depth){
            this.cnnInputSize = new int[]{height,width,depth};
            return this;
        }

        /** CNN input size, in order of {height,width,depth}.
         * @see #cnnInputSize(int, int, int)
         */
        public Builder cnnInputSize(int[] cnnInputSize){
            this.cnnInputSize = cnnInputSize;
            return this;
        }

        public MultiLayerConfiguration build() {

            //First: apply ConvolutionLayerSetup if necessary...
            if(cnnInputSize != null){
                new ConvolutionLayerSetup(this,cnnInputSize[0],cnnInputSize[1],cnnInputSize[2]);
            }

            //Second: apply layer preprocessors (dense/rnn layers etc) (a) where required, and (b) where no override is specified
            //ConvolutionLayerSetup should handle all CNN-related preprocessors; so just need to handle dense -> RNN etc
            int nLayers = this.confs.size();
            for( int i=1; i<nLayers; i++ ){
                if(inputPreProcessors.containsKey(i)) continue; //Override or ConvolutionLayerSetup

                Layer currLayer = this.confs.get(i).getLayer();
                if( currLayer instanceof ConvolutionLayer || currLayer instanceof SubsamplingLayer ) continue;  //Handled by ConvolutionLayerSetup
                Layer lastLayer = this.confs.get(i-1).getLayer();
                if(lastLayer instanceof ConvolutionLayer || lastLayer instanceof SubsamplingLayer ) continue; //Handled by ConvolutionLayerSetup
                //At this point: no CNN layers. must be dense/autoencoder/rbm etc or RNN
                if(currLayer instanceof DenseLayer || currLayer instanceof BasePretrainNetwork || currLayer instanceof OutputLayer ){
                    if(lastLayer instanceof BaseRecurrentLayer ){   //RNN -> FF
                        inputPreProcessors.put(i,new RnnToFeedForwardPreProcessor());
                    }
                } else if( currLayer instanceof BaseRecurrentLayer || currLayer instanceof RnnOutputLayer ){
                    if(lastLayer instanceof DenseLayer || lastLayer instanceof BasePretrainNetwork ){   //FF -> RNN
                        inputPreProcessors.put(i,new FeedForwardToRnnPreProcessor());
                    }
                }
            }


            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            conf.pretrain = pretrain;
            conf.backprop = backprop;
            conf.inputPreProcessors = inputPreProcessors;
            conf.backpropType = backpropType;
            conf.tbpttFwdLength = tbpttFwdLength;
            conf.tbpttBackLength = tbpttBackLength;
            conf.redistributeParams = redistributeParams;
            Nd4j.getRandom().setSeed(conf.getConf(0).getSeed());
            return conf;

        }


    }
}
