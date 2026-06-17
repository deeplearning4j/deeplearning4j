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
public class MelSpectrogram extends DynamicCustomOp {
    private int sampleRate = 22050;
    private int fftSize = 2048;
    private int hopLength = 512;
    private int numMelBins = 128;
    private double lowerEdgeHz = 0.0;
    private double upperEdgeHz = 8000.0;
    private double power = 2.0;

    public MelSpectrogram(@NonNull SameDiff sameDiff, @NonNull SDVariable input,
                           int sampleRate, int fftSize, int hopLength, int numMelBins,
                           double lowerEdgeHz, double upperEdgeHz, double power) {
        super(sameDiff, new SDVariable[]{input});
        this.sampleRate = sampleRate;
        this.fftSize = fftSize;
        this.hopLength = hopLength;
        this.numMelBins = numMelBins;
        this.lowerEdgeHz = lowerEdgeHz;
        this.upperEdgeHz = upperEdgeHz;
        this.power = power;
        addArgs();
    }

    public MelSpectrogram(@NonNull INDArray input, int sampleRate, int fftSize, int hopLength,
                           int numMelBins, double lowerEdgeHz, double upperEdgeHz, double power) {
        super(new INDArray[]{input}, null);
        this.sampleRate = sampleRate;
        this.fftSize = fftSize;
        this.hopLength = hopLength;
        this.numMelBins = numMelBins;
        this.lowerEdgeHz = lowerEdgeHz;
        this.upperEdgeHz = upperEdgeHz;
        this.power = power;
        addArgs();
    }

    @Override
    public String opName() {
        return "mel_spectrogram";
    }

    protected void addArgs() {
        addIArgument(sampleRate);
        addIArgument(fftSize);
        addIArgument(hopLength);
        addIArgument(numMelBins);
        addTArgument(lowerEdgeHz);
        addTArgument(upperEdgeHz);
        addTArgument(power);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for mel_spectrogram not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "Expected at least 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
