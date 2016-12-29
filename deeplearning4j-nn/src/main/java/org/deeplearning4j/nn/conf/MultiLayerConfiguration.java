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

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Configuration for a multi layer network
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
@Slf4j
public class MultiLayerConfiguration implements Serializable, Cloneable {

    protected List<NeuralNetConfiguration> confs;
    protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected boolean pretrain = false;
    protected boolean backprop = true;
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;

    //Counter for the number of parameter updates so far
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toYaml() {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        try {
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link MultiLayerConfiguration}
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
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromJson(String json) {
        MultiLayerConfiguration conf;
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            conf = mapper.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        //To maintain backward compatibility after loss function refactoring (configs generated with v0.5.0 or earlier)
        // Previously: enumeration used for loss functions. Now: use classes
        // IN the past, could have only been an OutputLayer or RnnOutputLayer using these enums
        int layerCount = 0;
        JsonNode confs = null;
        for(NeuralNetConfiguration nnc : conf.getConfs()){
            Layer l = nnc.getLayer();
            if(l instanceof BaseOutputLayer && ((BaseOutputLayer)l).getLossFn() == null){
                //lossFn field null -> may be an old config format, with lossFunction field being for the enum
                //if so, try walking the JSON graph to extract out the appropriate enum value

                BaseOutputLayer ol = (BaseOutputLayer)l;
                try{
                    JsonNode jsonNode = mapper.readTree(json);
                    if(confs == null) {
                        confs = jsonNode.get("confs");
                    }
                    if(confs instanceof ArrayNode){
                        ArrayNode layerConfs = (ArrayNode)confs;
                        JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                        if(outputLayerNNCNode == null) return conf; //Should never happen...
                        JsonNode outputLayerNode = outputLayerNNCNode.get("layer");

                        JsonNode lossFunctionNode = null;
                        if(outputLayerNode.has("output")){
                            lossFunctionNode = outputLayerNode.get("output").get("lossFunction");
                        } else if(outputLayerNode.has("rnnoutput")){
                            lossFunctionNode = outputLayerNode.get("rnnoutput").get("lossFunction");
                        }

                        if(lossFunctionNode != null){
                            String lossFunctionEnumStr = lossFunctionNode.asText();
                            LossFunctions.LossFunction lossFunction = null;
                            try{
                                lossFunction = LossFunctions.LossFunction.valueOf(lossFunctionEnumStr);
                            } catch (Exception e){
                                log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON",e);
                            }

                            if(lossFunction != null) {
                                switch (lossFunction) {
                                    case MSE:
                                        ol.setLossFn(new LossMSE());
                                        break;
                                    case XENT:
                                        ol.setLossFn(new LossBinaryXENT());
                                        break;
                                    case NEGATIVELOGLIKELIHOOD:
                                        ol.setLossFn(new LossNegativeLogLikelihood());
                                        break;
                                    case MCXENT:
                                        ol.setLossFn(new LossMCXENT());
                                        break;

                                    //Remaining: TODO
                                    case EXPLL:
                                    case RMSE_XENT:
                                    case SQUARED_LOSS:
                                    case RECONSTRUCTION_CROSSENTROPY:
                                    case CUSTOM:
                                    default:
                                        log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not set loss function for {}",lossFunction);
                                        break;
                                }
                            }
                        }

                    } else {
                        log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON: layer 'confs' field is not an ArrayNode (is: {})",
                            (confs != null ? confs.getClass() : null));
                    }
                } catch(IOException e){
                    log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON",e);
                    break;
                }
            }

            //Also, pre 0.7.2: activation functions were Strings ("activationFunction" field), not classes ("activationFn")
            //Try to load the old format if necessary, and create the appropriate IActivation instance
            if(l.getActivationFn() == null){
                try {
                    JsonNode jsonNode = mapper.readTree(json);
                    if(confs == null) {
                        confs = jsonNode.get("confs");
                    }
                    if (confs instanceof ArrayNode) {
                        ArrayNode layerConfs = (ArrayNode)confs;
                        JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                        if(outputLayerNNCNode == null) return conf; //Should never happen...
                        JsonNode layerWrapperNode = outputLayerNNCNode.get("layer");

                        if(layerWrapperNode == null || layerWrapperNode.size() != 1){
                            continue;
                        }

                        JsonNode layerNode = layerWrapperNode.elements().next();
                        JsonNode activationFunction = layerNode.get("activationFunction");      //Should only have 1 element: "dense", "output", etc

                        if(activationFunction != null){
                            IActivation ia = Activation.fromString(activationFunction.asText()).getActivationFunction();
                            l.setActivationFn(ia);
                        }
                    }

                } catch (IOException e){
                    log.warn("Layer with null ActivationFn field or pre-0.7.2 activation function detected: could not parse JSON",e);
                }
            }

            layerCount++;
        }
        return conf;
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
        protected double dampingFactor = 100;
        protected Map<Integer,InputPreProcessor> inputPreProcessors = new HashMap<>();
        protected boolean pretrain = false;
        protected boolean backprop = true;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = 20;
        protected int tbpttBackLength = 20;
        protected InputType inputType;
        @Deprecated
        protected int[] cnnInputSize;

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
         * @deprecated use {@link #setInputType(InputType)} with {@code InputType.convolutional(height,width,depth)}, for CNN data with
         * shape [minibatchSize,depth,height,width]. For image data that has been flattened into a row vector per example
         * (shape [minibatchSize,depth*height*width]) instead use {@code InputType.convolutionalFlat(height,width,depth)}
         */
        @Deprecated
        public Builder cnnInputSize(int height, int width, int depth){
            this.cnnInputSize = new int[]{height,width,depth};
            return this;
        }

        /** CNN input size, in order of {height,width,depth}.
         * @see #cnnInputSize(int, int, int)
         * @deprecated use {@link #setInputType(InputType)} with {@code InputType.convolutional(height,width,depth)}, for CNN data with
         * shape [minibatchSize,depth,height,width]. For image data that has been flattened into a row vector per example
         * (shape [minibatchSize,depth*height*width]) instead use {@code InputType.convolutionalFlat(height,width,depth)}
         */
        @Deprecated
        public Builder cnnInputSize(int[] cnnInputSize){
            if(cnnInputSize != null) {
                this.cnnInputSize = cnnInputSize;
            }
            return this;
        }

        public Builder setInputType(InputType inputType){
            this.inputType = inputType;
            return this;
        }

        public MultiLayerConfiguration build() {
            if (cnnInputSize != null) {
                new ConvolutionLayerSetup(this, cnnInputSize[0], cnnInputSize[1], cnnInputSize[2]);
            } else if (inputType == null && inputPreProcessors.get(0) == null) {
                //User hasn't set the InputType. Sometimes we can infer it...
                // For example, Dense/RNN layers, where preprocessor isn't set -> user is *probably* going to feed in
                // standard feedforward or RNN data
                //This isn't the most elegant implementation, but should avoid breaking backward compatibility here
                //Can't infer InputType for CNN layers, however (don't know image dimensions/depth)
                Layer firstLayer = confs.get(0).getLayer();
                if (firstLayer instanceof BaseRecurrentLayer) {
                    BaseRecurrentLayer brl = (BaseRecurrentLayer) firstLayer;
                    int nIn = brl.getNIn();
                    if (nIn > 0) {
                        inputType = InputType.recurrent(nIn);
                    }
                } else if (firstLayer instanceof DenseLayer ||
                    firstLayer instanceof EmbeddingLayer ||
                    firstLayer instanceof OutputLayer) {
                    //Can't just use "instanceof FeedForwardLayer" here. ConvolutionLayer is also a FeedForwardLayer
                    FeedForwardLayer ffl = (FeedForwardLayer) firstLayer;
                    int nIn = ffl.getNIn();
                    if (nIn > 0) {
                        inputType = InputType.feedForward(nIn);
                    }
                }
            }


            //Add preprocessors and set nIns, if InputType has been set
            // Builder.inputType field can be set in 1 of 4 ways:
            // 1. User calls setInputType directly
            // 2. Via ConvolutionLayerSetup -> internally calls setInputType(InputType.convolutional(...))
            // 3. User calls one of  the two cnnInputSize methods -> sets inputType field directly
            // 4. Via the above code: i.e., assume input is as expected  by the RNN or dense layer -> sets the inputType field
            if (inputType != null) {
                InputType currentInputType = inputType;
                for (int i = 0; i < confs.size(); i++) {
                    Layer l = confs.get(i).getLayer();
                    if (inputPreProcessors.get(i) == null) {
                        //Don't override preprocessor setting, but set preprocessor if required...
                        InputPreProcessor inputPreProcessor = l.getPreProcessorForInputType(currentInputType);
                        if (inputPreProcessor != null) {
                            inputPreProcessors.put(i, inputPreProcessor);
                        }
                    }

                    InputPreProcessor inputPreProcessor = inputPreProcessors.get(i);
                    if (inputPreProcessor != null) {
                        currentInputType = inputPreProcessor.getOutputType(currentInputType);
                    }
                    l.setNIn(currentInputType, false);  //Don't override the nIn setting, if it's manually set by the user

                    currentInputType = l.getOutputType(i, currentInputType);
                }

            }
            // Sets pretrain on the layer to track update for that specific layer
            if (isPretrain()) {
                for (int j = 0; j < confs.size(); j++) {
                    Layer l = confs.get(j).getLayer();
                    if (l instanceof BasePretrainNetwork) confs.get(j).setPretrain(pretrain);
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
            Nd4j.getRandom().setSeed(conf.getConf(0).getSeed());
            return conf;

        }


    }
}