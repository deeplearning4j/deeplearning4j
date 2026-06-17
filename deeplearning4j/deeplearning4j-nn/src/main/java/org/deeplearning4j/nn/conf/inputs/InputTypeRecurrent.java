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

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.nd4j.shade.jackson.annotation.JsonProperty;

@NoArgsConstructor
@Getter
@EqualsAndHashCode(callSuper = false)
public class InputTypeRecurrent extends InputType {
    private long size;
    private long timeSeriesLength;
    private RNNFormat format = RNNFormat.NCW;

    public InputTypeRecurrent(long size) {
        this(size, -1);
    }

    public InputTypeRecurrent(long size, long timeSeriesLength){
        this(size, timeSeriesLength, RNNFormat.NCW);
    }

    public InputTypeRecurrent(long size, RNNFormat format){
        this(size, -1, format);
    }

    public InputTypeRecurrent(@JsonProperty("size") long size,
                              @JsonProperty("timeSeriesLength") long timeSeriesLength,
                              @JsonProperty("format") RNNFormat format) {
        this.size = size;
        this.timeSeriesLength = timeSeriesLength;
        this.format = format;
    }

    @Override
    public Type getType() {
        return Type.RNN;
    }

    @Override
    public String toString() {
        if (timeSeriesLength > 0) {
            return "InputTypeRecurrent(" + size + ",timeSeriesLength=" + timeSeriesLength + ",format=" + format + ")";
        } else {
            return "InputTypeRecurrent(" + size + ",format=" + format + ")";
        }
    }

    @Override
    public long arrayElementsPerExample() {
        if (timeSeriesLength <= 0) {
            throw new IllegalStateException("Cannot calculate number of array elements per example: "
                    + "time series length is not set. Use InputType.recurrent(int size, int timeSeriesLength) instead?");
        }
        return timeSeriesLength * size;
    }

    @Override
    public long[] getShape(boolean includeBatchDim) {
        if (includeBatchDim){
            if (format == RNNFormat.NCW) {
                return new long[]{-1, size, timeSeriesLength};
            }
            else{
                return new long[]{-1, timeSeriesLength, size};
            }
        }
        else{
            if (format == RNNFormat.NCW) {
                return new long[]{size, timeSeriesLength};
            }
            else{
                return new long[]{timeSeriesLength, size};
            }
        }
    }
}
