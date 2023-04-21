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

package org.nd4j.linalg.api.ops.impl.reduce.longer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;

import java.util.Collections;
import java.util.List;

public class CountNonZero extends BaseReduceLongOp {
    public CountNonZero(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable i_v, boolean keepDims) {
        super(sameDiff, i_v, keepDims);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims) {
        super(sameDiff, i_v, dimensions, keepDims);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2) {
        super(sameDiff, i_v, i_v2);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable input, long[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, boolean keepDims) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable input, long... dimensions) {
        super(sameDiff, input, dimensions);
    }


    public CountNonZero(INDArray x, long... dimensions) {
        super(x, dimensions);
    }

    public CountNonZero(INDArray x, INDArray z, long... dimensions) {
        super(x, z, dimensions);
    }

    public CountNonZero(INDArray x, INDArray y, INDArray z, long... dimensions) {
        super(x, y, z, dimensions);
    }

    public CountNonZero(SameDiff sameDiff) {
        super(sameDiff);
    }

    public CountNonZero(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CountNonZero() {
    }

    public CountNonZero(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    public CountNonZero(INDArray in, boolean keepDims, long[] dimensions) {
       super(in,keepDims,dimensions);
    }

    public CountNonZero(SameDiff sd, SDVariable in, boolean keepDims, long[] dimensions) {
        super(sd,in,dimensions,keepDims);
    }

    public CountNonZero(INDArray in, long[] dimensions, boolean keepDims) {
        super(in,keepDims,dimensions);
    }


    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "countNonZero";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

}
