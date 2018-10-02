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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * OldReverse op
 */
public class OldReverse extends BaseTransformOp {
    public OldReverse(SameDiff sameDiff, SDVariable i_v, int... dimensions) {
        super(sameDiff, i_v, false);
        this.dimensions = dimensions;
    }

    public OldReverse() {
    }

    public OldReverse(INDArray x, INDArray z) {
        super(x, z);
    }

    public OldReverse(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public OldReverse(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public OldReverse(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public OldReverse(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 70;
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public String opName() {
        return "old_reverse";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable ret = f().reverse(f1.get(0), dimensions);
        return Arrays.asList(ret);
    }
}
