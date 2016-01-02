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

package org.deeplearning4j.nn.conf.layers;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;

/**
 * LSTM recurrent net, based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class GravesBidirectionalLSTM extends BaseRecurrentLayer {

    private double forgetGateBiasInit;

    private GravesBidirectionalLSTM(Builder builder) {
    	super(builder);
        this.forgetGateBiasInit = builder.forgetGateBiasInit;
    }

    @AllArgsConstructor @NoArgsConstructor
    public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

        private double forgetGateBiasInit = 1.0;

        /** Set forget gate bias initalizations. Values in range 1-5 can potentially
         * help with learning or longer-term dependencies.
         */
        public Builder forgetGateBiasInit(double biasInit){
            this.forgetGateBiasInit = biasInit;
            return this;
        }

        @SuppressWarnings("unchecked")
        public GravesBidirectionalLSTM build() {
            return new GravesBidirectionalLSTM(this);
        }
    }

}
