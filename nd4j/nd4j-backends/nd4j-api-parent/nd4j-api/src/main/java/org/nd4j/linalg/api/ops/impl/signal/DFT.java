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
 * Discrete Fourier Transform operation.
 * Computes the DFT of a complex input tensor.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@NoArgsConstructor
public class DFT extends DynamicCustomOp {
    private int axis = -2;
    private boolean inverse = false;
    private boolean onesided = false;

    public DFT(@NonNull SameDiff sameDiff, @NonNull SDVariable input, int axis, boolean inverse, boolean onesided) {
        super(sameDiff, new SDVariable[]{input});
        this.axis = axis;
        this.inverse = inverse;
        this.onesided = onesided;
        addArgs();
    }

    public DFT(@NonNull SameDiff sameDiff, @NonNull SDVariable input) {
        this(sameDiff, input, -2, false, false);
    }

    public DFT(@NonNull INDArray input, int axis, boolean inverse, boolean onesided, INDArray output) {
        super(new INDArray[]{input}, wrapOrNull(output));
        this.axis = axis;
        this.inverse = inverse;
        this.onesided = onesided;
        addArgs();
    }

    public DFT(@NonNull INDArray input, int axis, boolean inverse, boolean onesided) {
        this(input, axis, inverse, onesided, null);
    }

    @Override
    public String opName() {
        return "dft";
    }

    protected void addArgs() {
        addIArgument(axis);
        addIArgument(inverse ? 1 : 0);
        addIArgument(onesided ? 1 : 0);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for DFT not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() >= 1,
                "Expected at least 1 input datatype for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
