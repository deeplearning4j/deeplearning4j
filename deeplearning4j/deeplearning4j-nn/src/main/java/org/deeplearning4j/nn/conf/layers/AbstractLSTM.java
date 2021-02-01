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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class AbstractLSTM extends BaseRecurrentLayer {

    protected double forgetGateBiasInit;
    protected IActivation gateActivationFn = new ActivationSigmoid();
    protected boolean helperAllowFallback = true;

    protected AbstractLSTM(Builder builder) {
        super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
        this.gateActivationFn = builder.gateActivationFn;
        this.helperAllowFallback = builder.helperAllowFallback;
    }

    @AllArgsConstructor
    @NoArgsConstructor
    @Getter
    @Setter
    public static abstract class Builder<T extends Builder<T>> extends BaseRecurrentLayer.Builder<T> {

        /**
         * Set forget gate bias initalizations. Values in range 1-5 can potentially help with learning or longer-term
         * dependencies.
         */
        protected double forgetGateBiasInit = 1.0;

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         */
        protected IActivation gateActivationFn = new ActivationSigmoid();

        /**
         * When using CuDNN and an error is encountered, should fallback to the non-CuDNN implementatation be allowed?
         * If set to false, an exception in CuDNN will be propagated back to the user. If false, the built-in
         * (non-CuDNN) implementation for LSTM/GravesLSTM will be used
         *
         */
        protected boolean helperAllowFallback = true;

        /**
         * Set forget gate bias initalizations. Values in range 1-5 can potentially help with learning or longer-term
         * dependencies.
         */
        public T forgetGateBiasInit(double biasInit) {
            this.setForgetGateBiasInit(biasInit);
            return (T) this;
        }

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public T gateActivationFunction(String gateActivationFn) {
            return (T) gateActivationFunction(Activation.fromString(gateActivationFn));
        }

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public T gateActivationFunction(Activation gateActivationFn) {
            return (T) gateActivationFunction(gateActivationFn.getActivationFunction());
        }

        /**
         * Activation function for the LSTM gates. Note: This should be bounded to range 0-1: sigmoid or hard sigmoid,
         * for example
         *
         * @param gateActivationFn Activation function for the LSTM gates
         */
        public T gateActivationFunction(IActivation gateActivationFn) {
            this.setGateActivationFn(gateActivationFn);
            return (T) this;
        }

        /**
         * When using a helper (CuDNN or MKLDNN in some cases) and an error is encountered, should fallback to the non-helper implementation be allowed?
         * If set to false, an exception in the helper will be propagated back to the user. If false, the built-in
         * (non-helper) implementation for LSTM/GravesLSTM will be used
         *
         * @param allowFallback Whether fallback to non-helper implementation should be used
         */
        public T helperAllowFallback(boolean allowFallback) {
            this.setHelperAllowFallback(allowFallback);
            return (T) this;
        }

    }

}
