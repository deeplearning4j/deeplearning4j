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
public class MelFilterbank extends DynamicCustomOp {
    private int numMelBins = 128;
    private int fftSize = 2048;
    private int sampleRate = 22050;
    private double lowerEdgeHz = 0.0;
    private double upperEdgeHz = 8000.0;

    public MelFilterbank(@NonNull SameDiff sameDiff, int numMelBins, int fftSize, int sampleRate,
                          double lowerEdgeHz, double upperEdgeHz) {
        super(sameDiff, new SDVariable[]{});
        this.numMelBins = numMelBins;
        this.fftSize = fftSize;
        this.sampleRate = sampleRate;
        this.lowerEdgeHz = lowerEdgeHz;
        this.upperEdgeHz = upperEdgeHz;
        addArgs();
    }

    public MelFilterbank(int numMelBins, int fftSize, int sampleRate,
                          double lowerEdgeHz, double upperEdgeHz) {
        this(numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz, null);
    }

    public MelFilterbank(int numMelBins, int fftSize, int sampleRate,
                          double lowerEdgeHz, double upperEdgeHz, INDArray output) {
        super(new INDArray[]{}, wrapOrNull(output));
        this.numMelBins = numMelBins;
        this.fftSize = fftSize;
        this.sampleRate = sampleRate;
        this.lowerEdgeHz = lowerEdgeHz;
        this.upperEdgeHz = upperEdgeHz;
        addArgs();
    }

    @Override
    public String opName() {
        return "mel_filterbank";
    }

    protected void addArgs() {
        addIArgument(numMelBins);
        addIArgument(fftSize);
        addIArgument(sampleRate);
        addTArgument(lowerEdgeHz);
        addTArgument(upperEdgeHz);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for mel_filterbank not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(DataType.FLOAT);
    }
}
