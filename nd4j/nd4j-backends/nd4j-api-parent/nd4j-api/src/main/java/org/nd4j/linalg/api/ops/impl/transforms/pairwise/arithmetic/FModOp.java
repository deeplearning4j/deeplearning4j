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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * Floating-point mod
 *
 * @author raver119@gmail.com
 */
public class FModOp extends BaseTransformOp {
    public FModOp() {}

    public FModOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public FModOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public FModOp(SameDiff sameDiff) {
        super(sameDiff);
    }

    public FModOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public FModOp(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public FModOp(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public FModOp(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public FModOp(INDArray x, INDArray z) {
        super(x, z);
    }

    public FModOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public FModOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public FModOp(INDArray x) {
        super(x);
    }


    @Override
    public int opNum() {
        return 60;
    }

    @Override
    public String opName() {
        return "fmod";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException();
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return f().floorModBp(larg(), rarg(), f1.get(0));
    }
}
