/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.api.ops.impl.transforms.bool;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;

import java.util.Collections;
import java.util.List;

/**
 * Boolean NOT transform
 *
 * @author raver119@gmail.com
 */
public class BooleanNot extends BaseTransformBoolOp {

    public BooleanNot(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v, false);
    }

    public BooleanNot() {
        //
    }

    public BooleanNot(@NonNull INDArray x) {
        this(x, x);
    }

    public BooleanNot(@NonNull INDArray x, INDArray z) {
        super(x, null, z);
        Preconditions.checkArgument(x.dataType() == DataType.BOOL, "X operand must be BOOL");
        Preconditions.checkArgument(z.dataType() == DataType.BOOL, "Z operand must be BOOL");
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String onnxName() {
        return "not_applicable";
    }

    @Override
    public String tensorflowName() {
        return "not_applicable";
    }

    @Override
    public String opName() {
        return "bool_not";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(f().zerosLike(arg()));
    }
}
