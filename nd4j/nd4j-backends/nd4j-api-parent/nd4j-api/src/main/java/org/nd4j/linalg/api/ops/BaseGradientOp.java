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

package org.nd4j.linalg.api.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A gradient op always makes the following assumptions:
 * there is always a y (beacuse of backpropagating
 * or using the chain rule)
 *
 * and that it is special exec (for now)
 *
 * This op opType sis meant to be used
 * to build derivative operations.
 *
 *
 * @author Adam Gibson
 */
public abstract class BaseGradientOp extends BaseTransformOp implements GradientOp {
    public BaseGradientOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BaseGradientOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BaseGradientOp(INDArray x, INDArray z) {
        super(x, z);
        assertWrt();
    }

    public BaseGradientOp() {
    }

    public BaseGradientOp(INDArray x, INDArray z, long n) {
        super(x, z, n);
        assertWrt();
    }

    public BaseGradientOp(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        assertWrt();
    }

    public BaseGradientOp(INDArray x) {
        super(x);
        assertWrt();
    }

    /**
     * The array
     * to the gradient with respect to
     *
     * @return
     */
    @Override
    public INDArray wrt() {
        return y();
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public boolean isPassThrough() {
        return true;
    }

    private void assertWrt() {
        Preconditions.checkState(y != null,"A gradient op must define a wrt variable as a Y. ");
    }

}
