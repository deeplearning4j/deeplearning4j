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

package org.nd4j.linalg.api.ops.impl.reduce;

import lombok.EqualsAndHashCode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import java.util.List;

@EqualsAndHashCode
public class TensorMmulBp  extends DynamicCustomOp  {

    public TensorMmulBp(){}

    public TensorMmulBp(SameDiff samediff, SDVariable x, SDVariable y, SDVariable gradAtOutput, int[][] axes) {
        this(samediff, x, y, gradAtOutput, axes[0], axes[1] );
    }

    public TensorMmulBp(SameDiff samediff, SDVariable x, SDVariable y, SDVariable gradAtOutput, int[] axesX, int[] axesY ) {
        super(null, samediff, new SDVariable[]{x,y, gradAtOutput});
        int[][] axes = new int[][]{axesX, axesY};
        addIArgument(axesX.length);
        addIArgument(axesX);
        addIArgument(axesY.length);
        addIArgument(axesY);
    }

    public TensorMmulBp(INDArray x, INDArray y, INDArray gradAtOutput, int[][] axes) {
        this(x, y, gradAtOutput, axes[0], axes[1] );
    }

    public TensorMmulBp(INDArray x, INDArray y, INDArray gradAtOutput, int[] axesX, int[] axesY ) {
        super(null,new INDArray[]{x, y, gradAtOutput},null);
        int[][] axes = new int[][]{axesX, axesY};
        addIArgument(axesX.length);
        addIArgument(axesX);
        addIArgument(axesY.length);
        addIArgument(axesY);
    }

    public TensorMmulBp(INDArray x, INDArray y, INDArray gradAtOutput, INDArray dldx, INDArray dldy, int[][] axes ) {
        this(x, y, gradAtOutput, dldx, dldy, axes[0], axes[1] );
    }

    public TensorMmulBp(INDArray x, INDArray y, INDArray gradAtOutput, INDArray dldx, INDArray dldy, int[] axesX, int[] axesY  ) {
            super(null, new INDArray[]{x, y, gradAtOutput}, new INDArray[]{dldx, dldy});
            int[][] axes = new int[][]{axesX, axesY};
            addIArgument(axesX.length);
            addIArgument(axesX);
            addIArgument(axesY.length);
            addIArgument(axesY);
    }

    @Override
    public String opName() {
        return "tensormmul_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 inputs to tensormmul_bp op, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType() && dataTypes.get(0).isFPType(), "Inputs to tensormmul_bp op must both be a floating" +
                "point type: got %s", dataTypes);
        return dataTypes.subList(0, 2);
    }

}
