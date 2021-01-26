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

package org.nd4j.linalg.api.ops.impl.reduce.bp;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.Arrays;
import java.util.List;

@NoArgsConstructor
public class PowBp extends BaseDynamicTransformOp {

    public PowBp(SameDiff sameDiff, SDVariable x, SDVariable y, SDVariable dLdz) {
        super(sameDiff,new SDVariable[]{x,y,dLdz}, false);
    }

    public PowBp(INDArray x, INDArray y, INDArray dLdz,
                 INDArray dLdx, INDArray dLdy) {
        super(new INDArray[]{x,y, dLdz}, new INDArray[]{dLdx, dLdy});
    }

    @Override
    public String opName() {
        return "Pow_bp";
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got input %s", getClass(), dataTypes);
        //Gradient types: same as input
        return Arrays.asList(arg(0).dataType(), arg(1).dataType());
    }
}
