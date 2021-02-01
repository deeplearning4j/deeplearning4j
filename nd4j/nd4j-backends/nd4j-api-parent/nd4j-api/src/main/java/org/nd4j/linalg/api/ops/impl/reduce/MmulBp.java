/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.common.util.ArrayUtil;

import java.util.List;

/**
 * Matrix multiplication/dot product Backprop
 *
 * @author Paul Dubs
 */
@EqualsAndHashCode
public class MmulBp extends DynamicCustomOp {

    protected MMulTranspose mt;


    public MmulBp(SameDiff sameDiff,
                  SDVariable x,
                  SDVariable y,
                  SDVariable eps,
                  MMulTranspose mt) {
        super(null,sameDiff,new SDVariable[]{x,y,eps});
        this.mt = mt;
        addIArgument(ArrayUtil.fromBoolean(mt.isTransposeA()), ArrayUtil.fromBoolean(mt.isTransposeB()), ArrayUtil.fromBoolean(mt.isTransposeResult()));
    }


    public MmulBp(SameDiff sameDiff,
                  SDVariable x,
                  SDVariable y,
                  SDVariable eps) {
        this(sameDiff,x,y, eps, MMulTranspose.allFalse());
    }


    public MmulBp(INDArray x,
                  INDArray y,
                  INDArray eps,
                  INDArray dldx,
                  INDArray dldy,
                  MMulTranspose mt) {
        super(null, new INDArray[]{x, y, eps}, new INDArray[]{dldx, dldy});
        if (mt != null) {
          this.mt = mt;
          addIArgument(ArrayUtil.fromBoolean(mt.isTransposeA()),
                       ArrayUtil.fromBoolean(mt.isTransposeB()),
                       ArrayUtil.fromBoolean(mt.isTransposeResult()));
        }
    }


    public MmulBp() {}


    @Override
    public String opName() {
        return "matmul_bp";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        throw new UnsupportedOperationException("Differentiation of " + getClass().getName() + " not supported");
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 inputs to matmul_bp op, got %s", dataTypes);
        Preconditions.checkState(dataTypes.get(0).isFPType() && dataTypes.get(1).isFPType() && dataTypes.get(0).isFPType(), "Inputs to matmul_bp op must both be a floating" +
                "point type: got %s", dataTypes);
        return dataTypes.subList(0, 2);
    }
}

