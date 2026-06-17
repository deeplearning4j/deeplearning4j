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
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class InputTypeConvolutional3D extends InputType {
    private Convolution3D.DataFormat dataFormat;
    private long depth;
    private long height;
    private long width;
    private long channels;

    public InputTypeConvolutional3D(@JsonProperty("dataFormat") Convolution3D.DataFormat dataFormat,
                                    @JsonProperty("depth") long depth, @JsonProperty("height") long height, @JsonProperty("width") long width, @JsonProperty("channels") long channels) {
        this.dataFormat = dataFormat;
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.channels = channels;
    }

    @Override
    public Type getType() {
        return Type.CNN3D;
    }

    @Override
    public String toString() {
        return "InputTypeConvolutional3D(format=" + dataFormat + ",d=" + depth + ",h=" + height + ",w=" + width + ",c=" + channels + ")";
    }

    @Override
    public long arrayElementsPerExample() {
        return height * width * depth * channels;
    }

    @Override
    public long[] getShape(boolean includeBatchDim) {
        if(dataFormat == Convolution3D.DataFormat.NDHWC){
            if(includeBatchDim) return new long[]{-1, depth, height, width, channels};
            else return new long[]{depth, height, width, channels};
        } else {
            if(includeBatchDim) return new long[]{-1, channels, depth, height, width};
            else return new long[]{channels, depth, height, width};
        }
    }
}
