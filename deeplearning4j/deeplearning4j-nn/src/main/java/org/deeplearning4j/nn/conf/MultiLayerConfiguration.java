/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/



package org.deeplearning4j.nn.conf;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.memory.NetworkMemoryReport;
import org.deeplearning4j.util.OutputLayerUtil;
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
import java.util.*;

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
    protected Map<Integer, InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected boolean pretrain = false;
    protected boolean backprop = true;
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;

    @Getter
    @Setter
    protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;

    @Getter
    @Setter
    protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;

    @Getter
    @Setter
    protected CacheMode cacheMode;

    //Counter for the number of parameter updates so far
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    //Counter for the number of epochs completed so far. Used for per-epoch schedules
    protected int epochCount = 0;

    public int getEpochCount() {
        return epochCount;
    }

    public void setEpochCount(int epochCount) {
        this.epochCount = epochCount;
        for(int i = 0; i < confs.size(); i++) {
            getConf(i).setEpochCount(epochCount);
        }
    }

    /**
     *
     * @return  JSON representation of NN configuration
     */
    public String toYaml() {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        synchronized (mapper) {
            try {
                return mapper.writeValueAsString(this);
            } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
                throw new RuntimeException(e);
            }
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
        synchronized (mapper) {
            //JSON mappers are supposed to be thread safe: however, in practice they seem to miss fields occasionally
            //when writeValueAsString is used by multiple threads. This results in invalid JSON. See issue #3243
            try {
                return mapper.writeValueAsString(this);
            } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
                throw new RuntimeException(e);
            }
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
            //Check if this exception came from legacy legacy deserializer...
            String msg = e.getMessage();
            if(msg != null && msg.contains("legacy")){
                throw new RuntimeException("Error deserializing MultiLayerConfiguration - configuration may have a custom " +
                        "layer, vertex or preprocessor, in pre version 1.0.0-alpha JSON format. These layers can be " +
                        "deserialized by first registering them with NeuralNetConfiguration.registerLegacyCustomClassesForJSON(Class...)", e);
            }
            throw new RuntimeException(e);
        }


        //To maintain backward compatibility after loss function refactoring (configs generated with v0.5.0 or earlier)
        // Previously: enumeration used for loss functions. Now: use classes
        // IN the past, could have only been an OutputLayer or RnnOutputLayer using these enums
        int layerCount = 0;
        JsonNode confs = null;
        for (NeuralNetConfiguration nnc : conf.getConfs()) {
            Layer l = nnc.getLayer();
            if (l instanceof BaseOutputLayer && ((BaseOutputLayer) l).getLossFn() == null) {
                //lossFn field null -> may be an old config format, with lossFunction field being for the enum
                //if so, try walking the JSON graph to extract out the appropriate enum value

                BaseOutputLayer ol = (BaseOutputLayer) l;
                try {
                    JsonNode jsonNode = mapper.readTree(json);
                    if (confs == null) {
                        confs = jsonNode.get("confs");
                    }
                    if (confs instanceof ArrayNode) {
                        ArrayNode layerConfs = (ArrayNode) confs;
                        JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                        if (outputLayerNNCNode == null)
                            return conf; //Should never happen...
                        JsonNode outputLayerNode = outputLayerNNCNode.get("layer");

                        JsonNode lossFunctionNode = null;
                        if (outputLayerNode.has("output")) {
                            lossFunctionNode = outputLayerNode.get("output").get("lossFunction");
                        } else if (outputLayerNode.has("rnnoutput")) {
                            lossFunctionNode = outputLayerNode.get("rnnoutput").get("lossFunction");
                        }

                        if (lossFunctionNode != null) {
                            String lossFunctionEnumStr = lossFunctionNode.asText();
                            LossFunctions.LossFunction lossFunction = null;
                            try {
                                lossFunction = LossFunctions.LossFunction.valueOf(lossFunctionEnumStr);
                            } catch (Exception e) {
                                log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON",
                                                e);
                            }

                            if (lossFunction != null) {
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
                                        log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not set loss function for {}",
                                                        lossFunction);
                                        break;
                                }
                            }
                        }

                    } else {
                        log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON: layer 'confs' field is not an ArrayNode (is: {})",
                                        (confs != null ? confs.getClass() : null));
                    }
                } catch (IOException e) {
                    log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON",
                                    e);
                    break;
                }
            }

            //Also, pre 0.7.2: activation functions were Strings ("activationFunction" field), not classes ("activationFn")
            //Try to load the old format if necessary, and create the appropriate IActivation instance
            if ((l instanceof BaseLayer) && ((BaseLayer) l).getActivationFn() == null) {
                try {
                    JsonNode jsonNode = mapper.readTree(json);
                    if (confs == null) {
                        confs = jsonNode.get("confs");
                    }
                    if (confs instanceof ArrayNode) {
                        ArrayNode layerConfs = (ArrayNode) confs;
                        JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                        if (outputLayerNNCNode == null)
                            return conf; //Should never happen...
                        JsonNode layerWrapperNode = outputLayerNNCNode.get("layer");

                        if (layerWrapperNode == null || layerWrapperNode.size() != 1) {
                            continue;
                        }

                        JsonNode layerNode = layerWrapperNode.elements().next();
                        JsonNode activationFunction = layerNode.get("activationFunction"); //Should only have 1 element: "dense", "output", etc

                        if (activationFunction != null) {
                            IActivation ia = Activation.fromString(activationFunction.asText()).getActivationFunction();
                            ((BaseLayer) l).setActivationFn(ia);
                        }
                    }

                } catch (IOException e) {
                    log.warn("Layer with null ActivationFn field or pre-0.7.2 activation function detected: could not parse JSON",
                                    e);
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

            if (clone.confs != null) {
                List<NeuralNetConfiguration> list = new ArrayList<>();
                for (NeuralNetConfiguration conf : clone.confs) {
                    list.add(conf.clone());
                }
                clone.confs = list;
            }

            if (clone.inputPreProcessors != null) {
                Map<Integer, InputPreProcessor> map = new HashMap<>();
                for (Map.Entry<Integer, InputPreProcessor> entry : clone.inputPreProcessors.entrySet()) {
                    map.put(entry.getKey(), entry.getValue().clone());
                }
                clone.inputPreProcessors = map;
            }

            clone.inferenceWorkspaceMode = this.inferenceWorkspaceMode;
            clone.trainingWorkspaceMode = this.trainingWorkspaceMode;
            clone.cacheMode = this.cacheMode;

            return clone;

        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public InputPreProcessor getInputPreProcess(int curr) {
        return inputPreProcessors.get(curr);
    }

    /**
     * Get a {@link MemoryReport} for the given MultiLayerConfiguration. This is used to estimate the
     * memory requirements for the given network configuration and input
     *
     * @param inputType Input types for the network
     * @return Memory report for the network
     */
    public NetworkMemoryReport getMemoryReport(InputType inputType) {

        Map<String, MemoryReport> memoryReportMap = new LinkedHashMap<>();
        int nLayers = confs.size();
        for (int i = 0; i < nLayers; i++) {
            String layerName = confs.get(i).getLayer().getLayerName();
            if (layerName == null) {
                layerName = String.valueOf(i);
            }

            //Pass input type through preprocessor, if necessary
            InputPreProcessor preproc = getInputPreProcess(i);
            //TODO memory requirements for preprocessor
            if (preproc != null) {
                inputType = preproc.getOutputType(inputType);
            }

            LayerMemoryReport report = confs.get(i).getLayer().getMemoryReport(inputType);
            memoryReportMap.put(layerName, report);

            inputType = confs.get(i).getLayer().getOutputType(i, inputType);
        }

        return new NetworkMemoryReport(memoryReportMap, MultiLayerConfiguration.class, "MultiLayerNetwork", inputType);
    }

    /**
     * For the given input shape/type for the network, return a list of activation sizes for each layer in the network.<br>
     * i.e., list.get(i) is the output activation sizes for layer i
     * @param inputType Input type for the network
     * @return A lits of activation types for the network, indexed by layer number
     */
    public List<InputType> getLayerActivationTypes(@NonNull InputType inputType){
        List<InputType> out = new ArrayList<>();
        int nLayers = confs.size();
        for (int i = 0; i < nLayers; i++) {
            InputPreProcessor preproc = getInputPreProcess(i);
            if (preproc != null) {
                inputType = preproc.getOutputType(inputType);
            }

            inputType = confs.get(i).getLayer().getOutputType(i, inputType);
            out.add(inputType);
        }
        return out;
    }

    @Data
    public static class Builder {

        private static final int DEFAULT_TBPTT_LENGTH = 20;

        protected List<NeuralNetConfiguration> confs = new ArrayList<>();
        protected double dampingFactor = 100;
        protected Map<Integer, InputPreProcessor> inputPreProcessors = new HashMap<>();
        @Deprecated
        protected boolean pretrain = false;
        @Deprecated
        protected boolean backprop = true;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = DEFAULT_TBPTT_LENGTH;
        protected int tbpttBackLength = DEFAULT_TBPTT_LENGTH;
        protected InputType inputType;

        protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;
        protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;
        protected CacheMode cacheMode = CacheMode.NONE;
        protected boolean validateOutputConfig = true;

        /**
         * Specify the processors.
         * These are used at each layer for doing things like normalization and
         * shaping of input.
         * @param processor what to use to preProcess the data.
         * @return builder pattern
         */
        public Builder inputPreProcessor(Integer layer, InputPreProcessor processor) {
            inputPreProcessors.put(layer, processor);
            return this;
        }

        public Builder inputPreProcessors(Map<Integer, InputPreProcessor> processors) {
            this.inputPreProcessors = processors;
            return this;
        }

        /**
         * Whether to do back prop or not
         *
         * @deprecated doesn't affect training any more. Use {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#fit(DataSetIterator)} when training for backprop.
         *
         * @param backprop whether to do back prop or not
         * @return
         */
        @Deprecated
        public Builder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        /**
         * @deprecated Use {@link NeuralNetConfiguration.Builder#trainingWorkspaceMode(WorkspaceMode)}
         */
        @Deprecated
        public Builder trainingWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
            this.trainingWorkspaceMode = workspaceMode;
            return this;
        }

        /**
         * @deprecated Use {@link NeuralNetConfiguration.Builder#inferenceWorkspaceMode(WorkspaceMode)}
         */
        @Deprecated
        public Builder inferenceWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
            this.inferenceWorkspaceMode = workspaceMode;
            return this;
        }

        /**
         * This method defines how/if preOutput cache is handled:
         * NONE: cache disabled (default value)
         * HOST: Host memory will be used
         * DEVICE: GPU memory will be used (on CPU backends effect will be the same as for HOST)
         *
         * @param cacheMode
         * @return
         */
        public Builder cacheMode(@NonNull CacheMode cacheMode) {
            this.cacheMode = cacheMode;
            return this;
        }

        /**The type of backprop. Default setting is used for most networks (MLP, CNN etc),
         * but optionally truncated BPTT can be used for training recurrent neural networks.
         * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
         */
        public Builder backpropType(@NonNull BackpropType type) {
            this.backpropType = type;
            return this;
        }

        /**When doing truncated BPTT: how many steps should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * See: <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
         * @param bpttLength length > 0
         */
        public Builder tBPTTLength(int bpttLength) {
            tBPTTForwardLength(bpttLength);
            return tBPTTBackwardLength(bpttLength);
        }

        /**When doing truncated BPTT: how many steps of forward pass should we do
         * before doing (truncated) backprop?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
         * but may be larger than it in some circumstances (but never smaller)<br>
         * Ideally your training data time series length should be divisible by this
         * This is the k1 parameter on pg23 of
         * <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
         * @param forwardLength Forward length > 0, >= backwardLength
         */
        public Builder tBPTTForwardLength(int forwardLength) {
            this.tbpttFwdLength = forwardLength;
            return this;
        }

        /**When doing truncated BPTT: how many steps of backward should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * This is the k2 parameter on pg23 of
         * <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
         * @param backwardLength <= forwardLength
         */
        public Builder tBPTTBackwardLength(int backwardLength) {
            this.tbpttBackLength = backwardLength;
            return this;
        }

        /**
         * Whether to do pre train or not
         *
         * @deprecated doesn't affect training any more. Use {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#pretrain(DataSetIterator)} when training for layerwise pretraining.
         *
         * @param pretrain whether to do pre train or not
         * @return builder pattern
         */
        @Deprecated
        public Builder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public Builder confs(List<NeuralNetConfiguration> confs) {
            this.confs = confs;
            return this;
        }

        public Builder setInputType(InputType inputType) {
            this.inputType = inputType;
            return this;
        }

        /**
         * Enabled by default. If enabled, the output layer configuration will be validated, to throw an exception on
         * likely invalid outputs - such as softmax + nOut=1, or LossMCXENT + Tanh.<br>
         * If disabled (false) no output layer validation will be performed.<br>
         * Disabling this validation is not recommended, as the configurations that fail validation usually will
         * not be able to learn correctly. However, the option to disable this validation is provided for advanced users
         * when creating non-standard architectures.
         *
         * @param validate If true: validate output layer configuration. False: don't validate
         */
        public Builder validateOutputLayerConfig(boolean validate) {
            this.validateOutputConfig = validate;
            return this;
        }


        public MultiLayerConfiguration build() {
            //Validate BackpropType setting
            if((tbpttBackLength != DEFAULT_TBPTT_LENGTH || tbpttFwdLength != DEFAULT_TBPTT_LENGTH) && backpropType != BackpropType.TruncatedBPTT){
                log.warn("Truncated backpropagation through time lengths have been configured with values " + tbpttFwdLength
                        + " and " + tbpttBackLength + " but backprop type is set to " + backpropType + ". TBPTT configuration" +
                        " settings will only take effect if backprop type is set to BackpropType.TruncatedBPTT");
            }


            if (inputType == null && inputPreProcessors.get(0) == null) {
                //User hasn't set the InputType. Sometimes we can infer it...
                // For example, Dense/RNN layers, where preprocessor isn't set -> user is *probably* going to feed in
                // standard feedforward or RNN data
                //This isn't the most elegant implementation, but should avoid breaking backward compatibility here
                //Can't infer InputType for CNN layers, however (don't know image dimensions/depth)
                Layer firstLayer = confs.get(0).getLayer();
                if (firstLayer instanceof BaseRecurrentLayer) {
                    BaseRecurrentLayer brl = (BaseRecurrentLayer) firstLayer;
                    val nIn = brl.getNIn();
                    if (nIn > 0) {
                        inputType = InputType.recurrent(nIn);
                    }
                } else if (firstLayer instanceof DenseLayer || firstLayer instanceof EmbeddingLayer
                                || firstLayer instanceof OutputLayer) {
                    //Can't just use "instanceof FeedForwardLayer" here. ConvolutionLayer is also a FeedForwardLayer
                    FeedForwardLayer ffl = (FeedForwardLayer) firstLayer;
                    val nIn = ffl.getNIn();
                    if (nIn > 0) {
                        inputType = InputType.feedForward(nIn);
                    }
                }
            }


            //Add preprocessors and set nIns, if InputType has been set
            // Builder.inputType field can be set in 1 of 4 ways:
            // 1. User calls setInputType directly
            // 2. Via ConvolutionLayerSetup -> internally calls setInputType(InputType.convolutional(...))
            // 3. Via the above code: i.e., assume input is as expected  by the RNN or dense layer -> sets the inputType field
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
                    l.setNIn(currentInputType, false); //Don't override the nIn setting, if it's manually set by the user

                    currentInputType = l.getOutputType(i, currentInputType);
                }

            }
            // Sets pretrain on the layer to track update for that specific layer
            if (isPretrain()) {
                for (int j = 0; j < confs.size(); j++) {
                    Layer l = confs.get(j).getLayer();
                    if (l instanceof BasePretrainNetwork)
                        confs.get(j).setPretrain(pretrain);
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
            conf.trainingWorkspaceMode = trainingWorkspaceMode;
            conf.inferenceWorkspaceMode = inferenceWorkspaceMode;
            conf.cacheMode = cacheMode;

            Nd4j.getRandom().setSeed(conf.getConf(0).getSeed());

            //Validate output layer configuration
            if(validateOutputConfig) {
                //Validate output layer configurations...
                for(NeuralNetConfiguration n : conf.getConfs()){
                    Layer l = n.getLayer();
                    OutputLayerUtil.validateOutputLayer(l.getLayerName(), l); //No-op for non output/loss layers
                }
            }

            return conf;

        }


    }
}
