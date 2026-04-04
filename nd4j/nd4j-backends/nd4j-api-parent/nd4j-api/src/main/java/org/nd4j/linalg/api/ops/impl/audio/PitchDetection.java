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

package org.nd4j.linalg.api.ops.impl.audio;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class PitchDetection extends DynamicCustomOp {
    private int sampleRate = 22050;
    private int frameLength = 2048;
    private int hopLength = 512;
    private double minFreq = 80.0;
    private double maxFreq = 1000.0;

    public PitchDetection(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                           int sampleRate, int frameLength, int hopLength,
                           double minFreq, double maxFreq) {
        super(sameDiff, new SDVariable[]{input});
        this.sampleRate = sampleRate;
        this.frameLength = frameLength;
        this.hopLength = hopLength;
        this.minFreq = minFreq;
        this.maxFreq = maxFreq;
        addArgs();
    }

    public PitchDetection(@NonNull INDArray input, int sampleRate, int frameLength, int hopLength,
                           double minFreq, double maxFreq) {
        super(new INDArray[]{input}, null);
        this.sampleRate = sampleRate;
        this.frameLength = frameLength;
        this.hopLength = hopLength;
        this.minFreq = minFreq;
        this.maxFreq = maxFreq;
        addArgs();
    }

    @Override
    public String opName() {
        return "pitch_detection";
    }

    protected void addArgs() {
        addIArgument(sampleRate);
        addIArgument(frameLength);
        addIArgument(hopLength);
        addTArgument(minFreq);
        addTArgument(maxFreq);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for pitch_detection not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "Expected at least 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
