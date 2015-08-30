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

/**
 * Image LSTM recurrent net.
 * Based on karpathy et. al's work on generation of image descriptions.
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class ImageLSTM extends FeedForwardLayer {

    private ImageLSTM(Builder builder) {
    	super(builder);
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        
        @Override
        @SuppressWarnings("unchecked")
        public ImageLSTM build() {
            return new ImageLSTM(this);
        }
    }
}
