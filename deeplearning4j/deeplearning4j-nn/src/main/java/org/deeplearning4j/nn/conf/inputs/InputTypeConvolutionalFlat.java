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
import org.nd4j.shade.jackson.annotation.JsonProperty;

@Data
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class InputTypeConvolutionalFlat extends InputType {
    private long height;
    private long width;
    private long depth;

    public InputTypeConvolutionalFlat(@JsonProperty("height") long height, @JsonProperty("width") long width, @JsonProperty("depth") long depth) {
        this.height = height;
        this.width = width;
        this.depth = depth;
    }

    @Override
    public Type getType() {
        return Type.CNNFlat;
    }

    public long getFlattenedSize() {
        return height * width * depth;
    }

    public InputType getUnflattenedType() {
        return InputType.convolutional(height, width, depth);
    }

    @Override
    public String toString() {
        return "InputTypeConvolutionalFlat(h=" + height + ",w=" + width + ",d=" + depth + ")";
    }

    @Override
    public long arrayElementsPerExample() {
        return height * width * depth;
    }

    @Override
    public long[] getShape(boolean includeBatchDim) {
        if(includeBatchDim) return new long[]{-1, depth, height, width};
        else return new long[]{depth, height, width};
    }
}
