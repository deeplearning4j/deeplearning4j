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

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Boolean OR pairwise transform
 *
 * @author raver119@gmail.com
 */
public class Or extends BaseTransformOp {

    protected double comparable = 0.0;

    public Or(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
        this.extraArgs = new Object[] {this.comparable};
    }

    public Or(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
        this.extraArgs = new Object[] {this.comparable};
    }

    public Or(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double comparable) {
        super(sameDiff, i_v, inPlace);
        this.comparable = comparable;
        this.extraArgs = new Object[] {this.comparable};
    }

    public Or(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs, double comparable) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
        this.comparable = comparable;
        this.extraArgs = new Object[] {this.comparable};
    }

    public Or(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, double comparable) {
        super(sameDiff, i_v, extraArgs);
        this.comparable = comparable;
        this.extraArgs = new Object[] {this.comparable};
    }

    public Or() {}

    public Or(@NonNull INDArray x, @NonNull INDArray y) {
        this(x, y, 0.0);
    }

    public Or(@NonNull INDArray x, @NonNull INDArray y, Number comparable) {
        this(x, y, x, comparable, x.lengthLong());
    }

    public Or(@NonNull INDArray x, @NonNull INDArray y, INDArray z, Number comparable) {
        this(x, y, z, comparable, x.lengthLong());
    }

    public Or(@NonNull INDArray x, @NonNull INDArray y, long n) {
        this(x, y, x, n);
    }

    public Or(@NonNull INDArray x, @NonNull INDArray y, INDArray z) {
        this(x, y, z, z.lengthLong());
    }

    public Or(@NonNull INDArray x, @NonNull INDArray y, INDArray z, long n) {
        this(x, y, z, 0.0, n);
    }

    public Or(@NonNull INDArray x, @NonNull INDArray y, INDArray z, Number comparable, long n) {
        super(x, y, z, n);
        this.comparable = comparable.doubleValue();
        this.extraArgs = new Object[] {this.comparable};
    }


    @Override
    public int opNum() {
        return 57;
    }

    @Override
    public String opName() {
        return "boolean_or";
    }

    @Override
    public String onnxName() {
        return "Or";
    }

    @Override
    public String tensorflowName() {
        return "LogicalOr";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Arrays.asList( sameDiff.zerosLike(larg()), sameDiff.zerosLike(rarg()));
    }
}
