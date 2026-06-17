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

package org.nd4j.linalg.api.ops.impl.signal;

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

/**
 * Short-Time Fourier Transform operation.
 * Computes STFT by applying DFT to windowed overlapping segments.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class STFT extends DynamicCustomOp {
    private boolean onesided = true;

    public STFT(@NonNull SameDiff sameDiff, @NonNull SDVariable signal, @NonNull SDVariable frameStep,
                SDVariable window, SDVariable frameLength, boolean onesided) {
        super(sameDiff, buildInputs(signal, frameStep, window, frameLength));
        this.onesided = onesided;
        addArgs();
    }

    public STFT(@NonNull SameDiff sameDiff, @NonNull SDVariable signal, @NonNull SDVariable frameStep) {
        this(sameDiff, signal, frameStep, null, null, true);
    }

    public STFT(@NonNull SameDiff sameDiff, @NonNull SDVariable signal, @NonNull SDVariable frameStep, boolean onesided) {
        this(sameDiff, signal, frameStep, null, null, onesided);
    }

    public STFT(@NonNull INDArray signal, @NonNull INDArray frameStep, INDArray window, INDArray frameLength,
                boolean onesided, INDArray output) {
        super(buildInputs(signal, frameStep, window, frameLength), wrapOrNull(output));
        this.onesided = onesided;
        addArgs();
    }

    public STFT(@NonNull INDArray signal, @NonNull INDArray frameStep, INDArray window, INDArray frameLength,
                boolean onesided) {
        this(signal, frameStep, window, frameLength, onesided, null);
    }

    /**
     * Simplified STFT constructor without window/frameLength for codegen.
     */
    public STFT(@NonNull INDArray signal, @NonNull INDArray frameStep, boolean onesided) {
        this(signal, frameStep, null, null, onesided, null);
    }

    private static SDVariable[] buildInputs(SDVariable signal, SDVariable frameStep, SDVariable window, SDVariable frameLength) {
        if (window != null && frameLength != null) {
            return new SDVariable[]{signal, frameStep, window, frameLength};
        } else if (window != null) {
            return new SDVariable[]{signal, frameStep, window};
        } else {
            return new SDVariable[]{signal, frameStep};
        }
    }

    private static INDArray[] buildInputs(INDArray signal, INDArray frameStep, INDArray window, INDArray frameLength) {
        if (window != null && frameLength != null) {
            return new INDArray[]{signal, frameStep, window, frameLength};
        } else if (window != null) {
            return new INDArray[]{signal, frameStep, window};
        } else {
            return new INDArray[]{signal, frameStep};
        }
    }

    @Override
    public String opName() {
        return "stft";
    }

    protected void addArgs() {
        addIArgument(onesided ? 1 : 0);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for STFT not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 2,
                "Expected at least 2 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(DataType.FLOAT);
    }
}
