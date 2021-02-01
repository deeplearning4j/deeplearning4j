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

package org.nd4j.linalg.api.ops.impl.reduce.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceFloatOp;
import org.nd4j.linalg.api.ops.impl.reduce.bp.SumBp;

import java.util.Collections;
import java.util.List;

/**
 * Entropy Op - returns the entropy (information gain, or uncertainty of a random variable).
 * -sum(x * log(x))
 *
 * @author raver119@gmail.com
 */
public class Entropy extends BaseReduceFloatOp {
    public Entropy(SameDiff sameDiff, SDVariable i_v, int[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public Entropy() {}

    public Entropy(INDArray x, INDArray z, int... dimensions) {
        super(x, null, z, dimensions);
    }

    public Entropy(INDArray x, int... dimensions) {
        super(x, dimensions);
    }

    @Override
    public int opNum() {
        return 8;
    }

    @Override
    public String opName() {
        return "entropy";
    }

    @Override
    public Type getOpType() {
        return Type.REDUCE_FLOAT;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //dL/dx = dL/dOut * dOut/dIn
        //out = -sum(x*log(x))
        // let z = x * log(x)
        //Then we can do sumBp(z, -dL/dOut)
        //Note d/dx(x*log(x)) = log(x)+1

        return grad(sameDiff, arg(), f1.get(0), dimensions);
    }

    public static List<SDVariable> grad(SameDiff sd, SDVariable arg, SDVariable grad, int[] dimensions){
        SDVariable logx = sd.math.log(arg);
        SDVariable xLogX = arg.mul(logx);
        SDVariable sumBp = new SumBp(sd, xLogX, grad.neg(), false, dimensions).outputVariable();
        return Collections.singletonList(sumBp.mul(logx.add(1.0)));
    }
}
