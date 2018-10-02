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

package org.nd4j.linalg.api.ops.impl.transforms.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformFloatOp;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Arc Tangent elementwise function
 *
 * @author Adam Gibson
 */
public class ATan extends BaseTransformFloatOp {
    public ATan(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public ATan(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public ATan(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public ATan() {
    }

    public ATan(INDArray x, INDArray z) {
        super(x, z);
    }

    public ATan(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public ATan(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 11;
    }

    @Override
    public String opName() {
        return "atan";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Atan";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //d(atan(x))/dx = 1/(x^2+1)
        SDVariable xSqPlus1 = f().square(arg()).add(1.0);
        SDVariable ret = xSqPlus1.rdiv(1.0).mul(i_v.get(0));
        return Arrays.asList(ret);
    }
}
