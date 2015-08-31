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

import lombok.*;

import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * Output layer with different objective co-occurrences for different objectives.
 * This includes classification as well as prediction
 *
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class OutputLayer extends BaseOutputLayer {

    protected OutputLayer(Builder builder) {
    	super(builder);
    }

    @NoArgsConstructor
    public static class Builder extends BaseOutputLayer.Builder<Builder> {

        public Builder(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }
        
        @Override
        @SuppressWarnings("unchecked")
        public OutputLayer build() {
            return new OutputLayer(this);
        }
    }
}

