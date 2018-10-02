/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.api.ops.impl.transforms.same;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;

import java.util.Arrays;
import java.util.List;

/**
 * Negative function
 *
 * @author Adam Gibson
 */
public class Negative extends BaseTransformSameOp {
    public Negative(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Negative(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Negative(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Negative() {}

    public Negative(INDArray x, INDArray z) {
        super(x, z);
    }

    public Negative(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Negative(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String opName() {
        return "neg";
    }


    @Override
    public String onnxName() {
        return "Neg";
    }

    @Override
    public String tensorflowName() {
        return "Neg";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList(f().neg(i_v.get(0)));
    }



}
