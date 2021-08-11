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

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;

import java.util.Collections;
import java.util.List;

public class CountZero extends BaseReduceLongOp {

    public CountZero(SameDiff sameDiff, SDVariable input, int... dimensions) {
        super(sameDiff, input, dimensions);
    }


    public CountZero(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public CountZero(INDArray x, boolean keepDims, int... dimensions) {
        super(x, keepDims, dimensions);
    }

    public CountZero(INDArray x, INDArray z, int... dimensions) {
        super(x, z, dimensions);
    }

    public CountZero(INDArray x, INDArray y, INDArray z, int... dimensions) {
        super(x, y, z, dimensions);
    }

    public CountZero(SameDiff sameDiff) {
        super(sameDiff);
    }

    public CountZero() {
    }

    public CountZero(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    public CountZero(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public CountZero(SameDiff sameDiff, SDVariable input, int[] dimensions, boolean keepDims) {
        super(sameDiff, input, dimensions, keepDims);
    }

    public CountZero(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
    }

    public CountZero(SameDiff sameDiff, SDVariable i_v) {
        super(sameDiff, i_v);
    }

    public CountZero(SameDiff sd, SDVariable in, boolean keepDims, int[] dimensions) {
        super(sd,in,dimensions,keepDims);
    }


    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "countZero";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public String tensorflowName() {
        return "count_zero";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

}
