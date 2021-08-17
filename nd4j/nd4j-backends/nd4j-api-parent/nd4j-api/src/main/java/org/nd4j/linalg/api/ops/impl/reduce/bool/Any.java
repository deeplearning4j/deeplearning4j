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

package org.nd4j.linalg.api.ops.impl.reduce.bool;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceBoolOp;

import java.util.Collections;
import java.util.List;

public class Any extends BaseReduceBoolOp {
    public Any(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Any(INDArray x, INDArray z, boolean keepDims, int[] dimensions) {
        super(x, z, keepDims, dimensions);
    }

    public Any() {}

    public Any(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    public Any(INDArray x) {
        super(x);
    }

    public Any(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public Any(INDArray x, boolean keepDims, int... dimensions) {
        super(x, keepDims, dimensions);
    }

    public Any(INDArray x, INDArray z, int... dimensions) {
        super(x, z, dimensions);
    }

    public Any(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, dimensions);
    }

    public Any(SameDiff sameDiff) {
        super(sameDiff);
    }

    public Any(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Any(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public Any(SameDiff sameDiff, SDVariable i_v, boolean keepDims) {
        super(sameDiff, i_v, keepDims);
    }

    public Any(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public Any(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2) {
        super(sameDiff, i_v, i_v2);
    }

    public Any(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public Any(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
    }

    public Any(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "any";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public String onnxName() {
        return "Any";
    }

    @Override
    public String tensorflowName() {
        return "Any";
    }

    @Override
    public boolean emptyValue() {
        return false;
    }
}
