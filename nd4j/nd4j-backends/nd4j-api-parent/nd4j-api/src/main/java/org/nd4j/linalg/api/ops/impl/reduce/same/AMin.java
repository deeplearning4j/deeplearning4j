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

package org.nd4j.linalg.api.ops.impl.reduce.same;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceSameOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.MinBp;

import java.util.Collections;
import java.util.List;

/**
 * Calculate the absolute minimum over a vector
 *
 * @author raver119@gmail.com
 */
public class AMin extends BaseReduceSameOp {
    public AMin(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public AMin(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public AMin() {}

    public AMin(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    public AMin(INDArray x, INDArray z, int... dimensions) {
        super(x, null, z, dimensions);
    }


    @Override
    public int opNum() {
        return 6;
    }

    @Override
    public String opName() {
        return "amin";
    }


    @Override
    public Number getFinalResult() {
        return null;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        SDVariable sgn = sameDiff.math().sign(arg());
        SDVariable minBp = new MinBp(sameDiff, sameDiff.math().abs(arg()), f1.get(0), false, dimensions).outputVariable();
        return Collections.singletonList(sgn.mul(minBp));
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }
}
