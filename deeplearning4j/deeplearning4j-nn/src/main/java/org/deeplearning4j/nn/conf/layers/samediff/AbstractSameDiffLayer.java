/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.deeplearning4j.nn.conf.layers.samediff;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.SameDiffParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.NetworkUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

@Slf4j
@Data
@EqualsAndHashCode(callSuper = true)
public abstract class AbstractSameDiffLayer extends Layer {

    protected List<Regularization> regularization;
    protected List<Regularization> regularizationBias;
    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected GradientNormalization gradientNormalization;
    protected double gradientNormalizationThreshold = Double.NaN;

    private SDLayerParams layerParams;

    @Override
    public List<Regularization> getRegularizationByParam(String paramName) {
        if(layerParams.isWeightParam(paramName)){
            return regularization;
        } else if(layerParams.isBiasParam(paramName)){
            return regularizationBias;
        }
        return null;
    }

    protected AbstractSameDiffLayer(Builder builder) {
        super(builder);
        this.regularization = builder.regularization;
        this.regularizationBias = builder.regularizationBias;
        this.updater = builder.updater;
        this.biasUpdater = builder.biasUpdater;

        //Check that this class has a no-arg constructor for JSON: better to detect this now provide useful information
        // to pre-empt a failure later for users, which will have a more difficult to understand message
        try {
            getClass().getDeclaredConstructor();
        } catch (NoSuchMethodException e) {
            log.warn("***SameDiff layer {} does not have a zero argument (no-arg) constructor.***\nA no-arg constructor "
                            + "is required for JSON deserialization, which is used for both model saving and distributed (Spark) "
                            + "training.\nA no-arg constructor (private, protected or public) as well as setters (or simply a "
                            + "Lombok @Data annotation) should be added to avoid JSON errors later.",
                            getClass().getName());
        } catch (SecurityException e) {
            //Ignore
        }
    }

    protected AbstractSameDiffLayer() {
        //No op constructor for Jackson
    }

    public SDLayerParams getLayerParams() {
        if (layerParams == null) {
            layerParams = new SDLayerParams();
            defineParameters(layerParams);
        }
        return layerParams;
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //Default implementation: no-op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        //Default implementation: no-op
        return null;
    }


    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig) {
        //Default implementation: no op
    }

    /**
     * Define the parameters for the network. Use {@link SDLayerParams#addWeightParam(String, long...)} and {@link
     * SDLayerParams#addBiasParam(String, long...)}
     *
     * @param params Object used to set parameters for this layer
     */
    public abstract void defineParameters(SDLayerParams params);

    /**
     * Set the initial parameter values for this layer, if required
     *
     * @param params Parameter arrays that may be initialized
     */
    public abstract void initializeParameters(Map<String, INDArray> params);

    @Override
    public abstract org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                                Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                                boolean initializeParams, DataType networkDataType);

    //==================================================================================================================

    @Override
    public ParamInitializer initializer() {
        return SameDiffParamInitializer.getInstance();
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        if (biasUpdater != null && initializer().isBiasParam(this, paramName)) {
            return biasUpdater;
        } else if (initializer().isBiasParam(this, paramName) || initializer().isWeightParam(this, paramName)) {
            return updater;
        }
        throw new IllegalStateException("Unknown parameter key: " + paramName);
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return new LayerMemoryReport(); //TODO
    }

    /**
     * Returns the memory layout ('c' or 'f' order - i.e., row/column major) of the parameters. In most cases, this
     * can/should be left
     *
     * @param param Name of the parameter
     * @return Memory layout ('c' or 'f') of the parameter
     */
    public char paramReshapeOrder(String param) {
        return 'c';
    }

    protected void initWeights(int fanIn, int fanOut, WeightInit weightInit, INDArray array) {
        WeightInitUtil.initWeights(fanIn, fanOut, array.shape(), weightInit, null, paramReshapeOrder(null), array);
    }

    public void applyGlobalConfig(NeuralNetConfiguration.Builder b) {
        if (regularization == null || regularization.isEmpty()) {
            regularization = b.getRegularization();
        }
        if (regularizationBias == null || regularizationBias.isEmpty()) {
            regularizationBias = b.getRegularizationBias();
        }
        if (updater == null) {
            updater = b.getIUpdater();
        }
        if (biasUpdater == null) {
            biasUpdater = b.getBiasUpdater();
        }
        if (gradientNormalization == null) {
            gradientNormalization = b.getGradientNormalization();
        }
        if (Double.isNaN(gradientNormalizationThreshold)) {
            gradientNormalizationThreshold = b.getGradientNormalizationThreshold();
        }

        applyGlobalConfigToLayer(b);
    }

    @Getter
    @Setter
    public static abstract class Builder<T extends Builder<T>> extends Layer.Builder<T> {

        protected List<Regularization> regularization = new ArrayList<>();
        protected List<Regularization> regularizationBias = new ArrayList<>();

        /**
         * Gradient updater. For example, {@link org.nd4j.linalg.learning.config.Adam} or {@link
         * org.nd4j.linalg.learning.config.Nesterovs}
         *
         */
        protected IUpdater updater = null;

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as set by {@link
         * #updater(IUpdater)}
         *
         */
        protected IUpdater biasUpdater = null;

        /**
         * L1 regularization coefficient (weights only). Use {@link #l1Bias(double)} to configure the l1 regularization
         * coefficient for the bias.
         */
        public T l1(double l1) {
            //Check if existing L1 exists; if so, replace it
            NetworkUtils.removeInstances(this.regularization, L1Regularization.class);
            if(l1 > 0.0) {
                this.regularization.add(new L1Regularization(l1));
            }
            return (T) this;
        }

        /**
         * L2 regularization coefficient (weights only). Use {@link #l2Bias(double)} to configure the l2 regularization
         * coefficient for the bias.<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecay(double,boolean)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         */
        public T l2(double l2) {
            //Check if existing L2 exists; if so, replace it. Also remove weight decay - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularization, L2Regularization.class);
            if(l2 > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularization, WeightDecay.class, "WeightDecay regularization removed: incompatible with added L2 regularization");
                this.regularization.add(new L2Regularization(l2));
            }
            return (T) this;
        }

        /**
         * L1 regularization coefficient for the bias. Default: 0. See also {@link #l1(double)}
         */
        public T l1Bias(double l1Bias) {
            NetworkUtils.removeInstances(this.regularizationBias, L1Regularization.class);
            if(l1Bias > 0.0) {
                this.regularizationBias.add(new L1Regularization(l1Bias));
            }
            return (T) this;
        }

        /**
         * L2 regularization coefficient for the bias. Default: 0. See also {@link #l2(double)}<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecayBias(double,boolean)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         */
        public T l2Bias(double l2Bias) {
            NetworkUtils.removeInstances(this.regularizationBias, L2Regularization.class);
            if(l2Bias > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularizationBias, WeightDecay.class, "WeightDecay bias regularization removed: incompatible with added L2 regularization");
                this.regularizationBias.add(new L2Regularization(l2Bias));
            }
            return (T) this;
        }

        /**
         * Add weight decay regularization for the network parameters (excluding biases).<br>
         * This applies weight decay <i>with</i> multiplying the learning rate - see {@link WeightDecay} for more details.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @see #weightDecay(double, boolean)
         */
        public Builder weightDecay(double coefficient) {
            return weightDecay(coefficient, true);
        }

        /**
         * Add weight decay regularization for the network parameters (excluding biases). See {@link WeightDecay} for more details.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @param applyLR     Whether the learning rate should be multiplied in when performing weight decay updates. See {@link WeightDecay} for more details.
         * @see #weightDecay(double, boolean)
         */
        public Builder weightDecay(double coefficient, boolean applyLR) {
            //Check if existing weight decay if it exists; if so, replace it. Also remove L2 - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularization, WeightDecay.class);
            if(coefficient > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularization, L2Regularization.class, "L2 regularization removed: incompatible with added WeightDecay regularization");
                this.regularization.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Weight decay for the biases only - see {@link #weightDecay(double)} for more details.
         * This applies weight decay <i>with</i> multiplying the learning rate.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @see #weightDecayBias(double, boolean)
         */
        public Builder weightDecayBias(double coefficient) {
            return weightDecayBias(coefficient, true);
        }

        /**
         * Weight decay for the biases only - see {@link #weightDecay(double)} for more details<br>
         *
         * @param coefficient Weight decay regularization coefficient
         */
        public Builder weightDecayBias(double coefficient, boolean applyLR) {
            //Check if existing weight decay if it exists; if so, replace it. Also remove L2 - it doesn't make sense to use both
            NetworkUtils.removeInstances(this.regularizationBias, WeightDecay.class);
            if(coefficient > 0.0) {
                NetworkUtils.removeInstancesWithWarning(this.regularizationBias, L2Regularization.class, "L2 bias regularization removed: incompatible with added WeightDecay regularization");
                this.regularizationBias.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Set the regularization for the parameters (excluding biases) - for example {@link WeightDecay}<br>
         *
         * @param regularization Regularization to apply for the network parameters/weights (excluding biases)
         */
        public Builder regularization(List<Regularization> regularization) {
            this.setRegularization(regularization);
            return this;
        }

        /**
         * Set the regularization for the biases only - for example {@link WeightDecay}<br>
         *
         * @param regularizationBias Regularization to apply for the network biases only
         */
        public Builder regularizationBias(List<Regularization> regularizationBias) {
            this.setRegularizationBias(regularizationBias);
            return this;
        }

        /**
         * Gradient updater. For example, {@link org.nd4j.linalg.learning.config.Adam} or {@link
         * org.nd4j.linalg.learning.config.Nesterovs}
         *
         * @param updater Updater to use
         */
        public T updater(IUpdater updater) {
            this.setUpdater(updater);
            return (T) this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as set by {@link
         * #updater(IUpdater)}
         *
         * @param biasUpdater Updater to use for bias parameters
         */
        public T biasUpdater(IUpdater biasUpdater) {
            this.setBiasUpdater(biasUpdater);
            return (T) this;
        }
    }
}
