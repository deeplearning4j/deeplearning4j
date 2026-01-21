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

package org.nd4j.linalg.api.ops.impl.image;

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
 * AffineGrid operation - generates a 2D or 3D sampling grid from affine transformation matrices.
 * Used with GridSample for spatial transformer networks.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@NoArgsConstructor
public class AffineGrid extends DynamicCustomOp {
    private boolean alignCorners = false;

    public AffineGrid(@NonNull SameDiff sameDiff, @NonNull SDVariable theta, @NonNull SDVariable size,
                      boolean alignCorners) {
        super(sameDiff, new SDVariable[]{theta, size});
        this.alignCorners = alignCorners;
        addArgs();
    }

    public AffineGrid(@NonNull SameDiff sameDiff, @NonNull SDVariable theta, @NonNull SDVariable size) {
        this(sameDiff, theta, size, false);
    }

    public AffineGrid(@NonNull INDArray theta, @NonNull INDArray size, boolean alignCorners, INDArray output) {
        super(new INDArray[]{theta, size}, wrapOrNull(output));
        this.alignCorners = alignCorners;
        addArgs();
    }

    public AffineGrid(@NonNull INDArray theta, @NonNull INDArray size, boolean alignCorners) {
        this(theta, size, alignCorners, null);
    }

    @Override
    public String opName() {
        return "affine_grid";
    }

    protected void addArgs() {
        addIArgument(alignCorners ? 1 : 0);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for affine_grid not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2,
                "Expected 2 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(DataType.FLOAT);
    }
}
