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

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelpers;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@Deprecated
public class GravesBidirectionalLSTM extends BaseRecurrentLayer {

    private double forgetGateBiasInit;
    private IActivation gateActivationFn = new ActivationSigmoid();
    protected boolean helperAllowFallback = true;

    private GravesBidirectionalLSTM(Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
        this.gateActivationFn = builder.gateActivationFn;
        this.helperAllowFallback = builder.helperAllowFallback;

        initializeConstraints(builder);
    }

    @Override
    protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder) {
        super.initializeConstraints(builder);
        if (((Builder) builder).recurrentConstraints != null) {
            if (constraints == null) {
                constraints = new ArrayList<>();
            }
            for (LayerConstraint c : ((Builder) builder).recurrentConstraints) {
                LayerConstraint c2 = c.clone();
                Set<String> s = new HashSet<>();
                s.add(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS);
                s.add(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS);
                c2.setParams(s);
                constraints.add(c2);
            }
        }
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
        org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM ret =
                        new org.deeplearning4j.nn.layers.recurrent.GravesBidirectionalLSTM(conf, networkDataType);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return GravesBidirectionalLSTMParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return LSTMHelpers.getMemoryReport(this, inputType);
    }

    @AllArgsConstructor
    @NoArgsConstructor
    @Getter
    @Setter
    public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

        /**
         * Set forget gate bias initalizations. Values in range 1-5 can potentially help with learning or longer-term
         * dependencies.
         */
        private double forgetGateBiasInit = 1.0;

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         */
        private IActivation gateActivationFn = new ActivationSigmoid();

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for GravesBidirectionalLSTM will be used
         *
         */
        protected boolean helperAllowFallback = true;

        /**
         * Set forget gate bias initalizations. Values in range 1-5 can potentially help with learning or longer-term
         * dependencies.
         */
        public Builder forgetGateBiasInit(double biasInit) {
            this.setForgetGateBiasInit(biasInit);
            return this;
        }

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public Builder gateActivationFunction(String gateActivationFn) {
            return gateActivationFunction(Activation.fromString(gateActivationFn));
        }

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public Builder gateActivationFunction(Activation gateActivationFn) {
            return gateActivationFunction(gateActivationFn.getActivationFunction());
        }

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public Builder gateActivationFunction(IActivation gateActivationFn) {
            this.setGateActivationFn(gateActivationFn);
            return this;
        }

        /**
         * When using a helper (CuDNN or MKLDNN in some cases) and an error is encountered, should fallback to the non-helper implementation be allowed?
         * If set to false, an exception in the helper will be propagated back to the user. If false, the built-in
         * (non-helper) implementation for GravesBidirectionalLSTM will be used
         *
         * @param allowFallback Whether fallback to non-helper implementation should be used
         */
        public Builder helperAllowFallback(boolean allowFallback) {
            this.setHelperAllowFallback(allowFallback);
            return (Builder) this;
        }

        @SuppressWarnings("unchecked")
        public GravesBidirectionalLSTM build() {
            return new GravesBidirectionalLSTM(this);
        }
    }

}
