/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.api.ops.impl.reduce.longer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceLongOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.util.Collections;
import java.util.List;

/**
 * This operation returns number of elements matching specified condition
 *
 * @author raver119@gmail.com
 */
public class MatchCondition extends BaseReduceLongOp {

    private double compare;
    private double eps;
    private int mode;

    public MatchCondition(SameDiff sameDiff, SDVariable in, Condition condition) {
        this(sameDiff, in, condition, false, null);
    }

    public MatchCondition(SameDiff sameDiff, SDVariable in, Condition condition, boolean keepDims, int... dimensions) {
        super(sameDiff, in, dimensions, keepDims);
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = Nd4j.EPS_THRESHOLD;
        this.extraArgs = new Object[] {compare, eps, (double) mode};
    }

    public MatchCondition() {}

    public MatchCondition(INDArray x, Condition condition, int... dimensions) {
        this(x, Nd4j.EPS_THRESHOLD, condition, dimensions);
    }

    public MatchCondition(INDArray x, Condition condition, boolean keepDims, int... dimensions) {
        this(x, Nd4j.EPS_THRESHOLD, condition, dimensions);
        this.keepDims = keepDims;
    }

    public MatchCondition(INDArray x, double eps, Condition condition, int... dimensions) {
        super(x);
        this.compare = condition.getValue();
        this.mode = condition.condtionNum();
        this.eps = eps;

        this.extraArgs = new Object[] {compare, eps, (double) mode};

        defineDimensions(dimensions);
    }

    @Override
    public int opNum() {
        return 2;
    }

    @Override
    public String opName() {
        return "match_condition";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
