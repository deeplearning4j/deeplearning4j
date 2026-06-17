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

package org.deeplearning4j.nn.conf.inputs;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.nd4j.common.util.OneTimeLogger;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Slf4j
@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class InputTypeConvolutional extends InputType {
    private long height;
    private long width;
    private long channels;
    private CNN2DFormat format = CNN2DFormat.NCHW;

    public InputTypeConvolutional(@JsonProperty("height") long height, @JsonProperty("width") long width,
                                  @JsonProperty("channels") long channels, @JsonProperty("format") CNN2DFormat format) {
        if(height <= 0) {
            OneTimeLogger.warn(log,"Assigning height of 0. Normally this is not valid. Exceptions for this are generally related" +
                    "to model import and unknown dimensions");
        }

        if(width <= 0) {
            OneTimeLogger.warn(log,"Assigning width of 0. Normally this is not valid. Exceptions for this are generally related" +
                    "to model import and unknown dimensions");
        }


        if(channels <= 0) {
            OneTimeLogger.warn(log,"Assigning channels of 0. Normally this is not valid. Exceptions for this are generally related" +
                    "to model import and unknown dimensions");
        }


        this.height = height;
        this.width = width;
        this.channels = channels;
        if(format != null)
            this.format = format;
    }

    public InputTypeConvolutional(long height, long width, long channels) {
        this(height, width, channels, CNN2DFormat.NCHW);
    }

    @Deprecated
    public long getDepth() {
        return channels;
    }

    @Deprecated
    public void setDepth(long depth) {
        this.channels = depth;
    }

    @Override
    public Type getType() {
        return Type.CNN;
    }

    @Override
    public String toString() {
        return "InputTypeConvolutional(h=" + height + ",w=" + width + ",c=" + channels + "," + format + ")";
    }

    @Override
    public long arrayElementsPerExample() {
        return height * width * channels;
    }

    @Override
    public long[] getShape(boolean includeBatchDim) {
        if(format == CNN2DFormat.NCHW){
            if(includeBatchDim) return new long[]{-1, channels, height, width};
            else return new long[]{channels, height, width};
        } else {
            if(includeBatchDim) return new long[]{-1, height, width, channels};
            else return new long[]{height, width, channels};
        }
    }
}
