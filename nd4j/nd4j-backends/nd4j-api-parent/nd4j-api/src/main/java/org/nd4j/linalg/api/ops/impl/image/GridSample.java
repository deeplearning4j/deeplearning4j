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
 * GridSample operation - samples input tensor using grid coordinates.
 * Implements bilinear/nearest/bicubic interpolation with various padding modes.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class GridSample extends DynamicCustomOp {
    public enum Mode { BILINEAR, NEAREST, BICUBIC }
    public enum PaddingMode { ZEROS, BORDER, REFLECTION }

    private Mode mode = Mode.BILINEAR;
    private PaddingMode paddingMode = PaddingMode.ZEROS;
    private boolean alignCorners = false;

    public GridSample(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable grid,
                      @NonNull Mode mode, @NonNull PaddingMode paddingMode, boolean alignCorners) {
        super(sameDiff, new SDVariable[]{input, grid});
        this.mode = mode;
        this.paddingMode = paddingMode;
        this.alignCorners = alignCorners;
        addArgs();
    }

    public GridSample(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable grid) {
        this(sameDiff, input, grid, Mode.BILINEAR, PaddingMode.ZEROS, false);
    }

    public GridSample(@NonNull SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable grid,
                      @NonNull String mode, @NonNull String paddingMode, boolean alignCorners) {
        this(sameDiff, input, grid, parseMode(mode), parsePaddingMode(paddingMode), alignCorners);
    }

    private static Mode parseMode(String mode) {
        switch (mode.toLowerCase()) {
            case "bilinear": return Mode.BILINEAR;
            case "nearest": return Mode.NEAREST;
            case "bicubic": return Mode.BICUBIC;
            default: throw new IllegalArgumentException("Unknown mode: " + mode + ". Expected: bilinear, nearest, or bicubic");
        }
    }

    private static PaddingMode parsePaddingMode(String paddingMode) {
        switch (paddingMode.toLowerCase()) {
            case "zeros": return PaddingMode.ZEROS;
            case "border": return PaddingMode.BORDER;
            case "reflection": return PaddingMode.REFLECTION;
            default: throw new IllegalArgumentException("Unknown padding mode: " + paddingMode + ". Expected: zeros, border, or reflection");
        }
    }

    public GridSample(@NonNull INDArray input, @NonNull INDArray grid,
                      @NonNull Mode mode, @NonNull PaddingMode paddingMode, boolean alignCorners, INDArray output) {
        super(new INDArray[]{input, grid}, wrapOrNull(output));
        this.mode = mode;
        this.paddingMode = paddingMode;
        this.alignCorners = alignCorners;
        addArgs();
    }

    public GridSample(@NonNull INDArray input, @NonNull INDArray grid,
                      @NonNull Mode mode, @NonNull PaddingMode paddingMode, boolean alignCorners) {
        this(input, grid, mode, paddingMode, alignCorners, null);
    }

    @Override
    public String opName() {
        return "grid_sample";
    }

    protected void addArgs() {
        addIArgument(mode.ordinal());
        addIArgument(paddingMode.ordinal());
        addIArgument(alignCorners ? 1 : 0);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        throw new UnsupportedOperationException("Gradient for grid_sample not implemented");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2,
                "Expected 2 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
