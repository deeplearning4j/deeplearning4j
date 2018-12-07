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
 * Log elementwise function
 *
 * @author Adam Gibson
 */
public class ACos extends BaseTransformFloatOp {
    public ACos(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public ACos(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public ACos(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public ACos() {
    }

    public ACos(INDArray x) {
        super(x);
    }

    public ACos(INDArray x, INDArray y) {
        super(x, y);
    }

    public ACos(INDArray indArray, INDArray indArray1, int length) {
        super(indArray, indArray1, length);
    }

    @Override
    public int opNum() {
        return 9;
    }

    @Override
    public String opName() {
        return "acos";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Acos";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //dacos(x)/dx = -1 / sqrt(1-x^2)
        SDVariable oneSubSq = f().square(arg()).rsub(1.0);
        SDVariable sqrt = f().sqrt(oneSubSq);
        SDVariable ret = sqrt.rdiv(-1.0).mul(i_v.get(0));
        return Arrays.asList(ret);
    }
}
