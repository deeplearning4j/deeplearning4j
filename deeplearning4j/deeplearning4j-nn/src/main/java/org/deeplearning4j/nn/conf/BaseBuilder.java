/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.deeplearning4j.nn.conf;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Data
public abstract class BaseBuilder {

    protected static final int DEFAULT_TBPTT_LENGTH = 20;

    protected List<NeuralNetConfiguration> confs = new ArrayList<>();
    protected double dampingFactor = 100;
    protected Map<Integer, InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = DEFAULT_TBPTT_LENGTH;
    protected int tbpttBackLength = DEFAULT_TBPTT_LENGTH;
    protected InputType inputType;

    protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;
    protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;
    protected CacheMode cacheMode = CacheMode.NONE;
    protected boolean validateOutputConfig = true;
    protected boolean validateTbpttConfig = true;
    protected DataType dataType;
    protected boolean overrideNinUponBuild = true;


    /**
     * Whether to over ride the nIn
     * configuration forcibly upon construction.
     * Default value is true
     * @param overrideNinUponBuild Whether to over ride the nIn
     *           configuration forcibly upon construction.
     * @return builder pattern
     */
    public  <T extends BaseBuilder> T overrideNinUponBuild(boolean overrideNinUponBuild) {
        this.overrideNinUponBuild = overrideNinUponBuild;
        return  (T) this;
    }

    /**
     * Specify the processors.
     * These are used at each layer for doing things like normalization and
     * shaping of input.
     *
     * @param processor what to use to preProcess the data.
     * @return builder pattern
     */
    public  <T extends BaseBuilder> T inputPreProcessor(Integer layer, InputPreProcessor processor) {
        inputPreProcessors.put(layer, processor);
        return  (T) this;
    }

    public  <T extends BaseBuilder> T inputPreProcessors(Map<Integer, InputPreProcessor> processors) {
        this.inputPreProcessors = processors;
        return  (T) this;
    }

    /**
     * @deprecated Use {@link NeuralNetConfiguration.Builder#trainingWorkspaceMode(WorkspaceMode)}
     */
    @Deprecated
    public  <T extends BaseBuilder> T trainingWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
        this.trainingWorkspaceMode = workspaceMode;
        return  (T) this;
    }

    /**
     * @deprecated Use {@link NeuralNetConfiguration.Builder#inferenceWorkspaceMode(WorkspaceMode)}
     */
    @Deprecated
    public  <T extends BaseBuilder> T inferenceWorkspaceMode(@NonNull WorkspaceMode workspaceMode) {
        this.inferenceWorkspaceMode = workspaceMode;
        return (T) this;
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
    public  <T extends BaseBuilder> T cacheMode(@NonNull CacheMode cacheMode) {
        this.cacheMode = cacheMode;
        return  (T) this;
    }

    /**
     * The type of backprop. Default setting is used for most networks (MLP, CNN etc),
     * but optionally truncated BPTT can be used for training recurrent neural networks.
     * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
     */
    public  <T extends BaseBuilder> T backpropType(@NonNull BackpropType type) {
        this.backpropType = type;
        return  (T) this;
    }

    /**
     * When doing truncated BPTT: how many steps should we do?<br>
     * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
     * See: <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
     *
     * @param bpttLength length > 0
     */
    public  <T extends BaseBuilder> T tBPTTLength(int bpttLength) {
        tBPTTForwardLength(bpttLength);
        return tBPTTBackwardLength(bpttLength);
    }

    /**
     * When doing truncated BPTT: how many steps of forward pass should we do
     * before doing (truncated) backprop?<br>
     * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
     * Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
     * but may be larger than it in some circumstances (but never smaller)<br>
     * Ideally your training data time series length should be divisible by this
     * This is the k1 parameter on pg23 of
     * <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
     *
     * @param forwardLength Forward length > 0, >= backwardLength
     */
    public  <T extends BaseBuilder> T tBPTTForwardLength(int forwardLength) {
        this.tbpttFwdLength = forwardLength;
        return (T) this;
    }

    /**
     * When doing truncated BPTT: how many steps of backward should we do?<br>
     * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
     * This is the k2 parameter on pg23 of
     * <a href="http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf">http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf</a>
     *
     * @param backwardLength <= forwardLength
     */
    public  <T extends BaseBuilder> T tBPTTBackwardLength(int backwardLength) {
        this.tbpttBackLength = backwardLength;
        return (T) this;
    }

    public  <T extends BaseBuilder> T confs(List<NeuralNetConfiguration> confs) {
        this.confs = confs;
        return (T) this;
    }

    public  <T extends BaseBuilder> T setInputType(InputType inputType) {
        this.inputType = inputType;
        return (T) this;
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
    public <T extends BaseBuilder> T validateOutputLayerConfig(boolean validate) {
        this.validateOutputConfig = validate;
        return (T) this;
    }

    /**
     * Enabled by default. If enabled, an exception will be throw when using the (invalid) combination of truncated
     * backpropagation through time (TBPTT) with either a GlobalPoolingLayer or LastTimeStepLayer.<br>
     * It is possible to disable this validation to allow what is almost certainly an invalid configuration to be used,
     * however this is not recommended.
     *
     * @param validate Whether TBPTT validation should be performed
     */
    public <T extends BaseBuilder> T  validateTbpttConfig(boolean validate){
        this.validateTbpttConfig = validate;
        return (T) this;
    }

    /**
     * Set the DataType for the network parameters and activations for all layers in the network. Default: Float
     * @param dataType Datatype to use for parameters and activations
     */
    public <T extends BaseBuilder> T  dataType(@NonNull DataType dataType) {
        this.dataType = dataType;
        return (T) this;
    }

    public abstract <T> T build();


}
