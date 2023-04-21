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

package org.nd4j.linalg.api.ops.impl.reduce3;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class EqualsWithEps extends BaseReduce3Op {
    private double eps;

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, long[] dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions) {
        super(sameDiff, i_v, dimensions);
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, long[] dimensions, double eps) {
        super(sameDiff, i_v, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double eps, long... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, double eps) {
        super(sameDiff, i_v, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double eps) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(double eps) {
        this.eps = eps;
    }

    public EqualsWithEps(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, double eps) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps() {}

    public EqualsWithEps(INDArray x, INDArray y, long... dimensions) {
        super(x, y, dimensions);
        this.eps = Nd4j.EPS_THRESHOLD;
    }

    public EqualsWithEps(INDArray x, INDArray y, boolean allDistances, long... dimensions) {
        super(x, y, allDistances, dimensions);
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, double eps, long... dimensions) {
        super(x, y, z, false, dimensions);
        this.extraArgs = new Object[] {0.0, 0.0, eps};
    }

    public EqualsWithEps(INDArray x, INDArray y, double eps, long... dimensions) {
        this(x, y, null, eps, dimensions);
    }

    public EqualsWithEps(INDArray x, INDArray y, boolean allDistances, double eps, long... dimensions) {
        super(x, y, allDistances, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, double eps) {
        super(x, y, z);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, boolean keepDims, double eps, long... dimensions) {
        super(x, y, z, keepDims, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, double eps, long... dimensions) {
        super(x, y, z, keepDims, allDistances, dimensions);
        this.eps = eps;
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z) {
        this(x, y, z, Nd4j.EPS_THRESHOLD, null);
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, boolean keepDims, long... dimensions) {
        super(x, y, z, keepDims, dimensions);
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, boolean keepDims, boolean allDistances, long... dimensions) {
        super(x, y, z, keepDims, allDistances, dimensions);
    }

    public EqualsWithEps(INDArray x, INDArray y, INDArray z, long... dimensions) {
        super(x, y, z, dimensions);
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "equals_with_eps";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(outputVariables()[0]);
    }
}
