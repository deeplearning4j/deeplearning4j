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

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.CubeDerivative;

import java.util.Collections;
import java.util.List;

/**
 * Rectified linear unit 6, i.e. min(max(input, cutoff), 6), where cutoff can be chosen.
 *
 * @author Max Pumperla
 */
public class Relu6 extends BaseTransformOp {

    private double cutoff = 0.0;

    public Relu6(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, double cutoff) {
        super(sameDiff, i_v1, i_v2, inPlace);
        this.cutoff = cutoff;
        this.extraArgs = new Object[]{cutoff};
    }

    public Relu6(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs, double cutoff) {
        super(sameDiff, i_v1, i_v2, extraArgs);
        this.cutoff = cutoff;
        this.extraArgs = new Object[]{cutoff};
    }

    public Relu6(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, inPlace);
        this.cutoff = cutoff;
        this.extraArgs = new Object[]{cutoff};

    }

    public Relu6() {
        this.extraArgs = new Object[]{cutoff};
    }

    public Relu6(INDArray x, INDArray z, double cutoff) {
        super(x, z);
        this.cutoff = cutoff;
        init(x, y, z, n); //Need to re-init to properly set cutoff in extra args array
    }

    public Relu6(INDArray x, INDArray z, long n, double cutoff) {
        super(x, z, n);
        this.cutoff = cutoff;
        init(x, y, z, n);
    }

    public Relu6(INDArray x, INDArray y, INDArray z, long n, double cutoff) {
        super(x, y, z, n);
        this.cutoff = cutoff;
        init(x, y, z, n);
    }

    public Relu6(INDArray x, double cutoff) {
        super(x);
        this.cutoff = cutoff;
        init(x, y, z, n);
    }

    public Relu6(INDArray x, INDArray z) {
        super(x, z);
    }

    public Relu6(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Relu6(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Relu6(INDArray x, INDArray y, INDArray z) {
        super(x, y, z, x.lengthLong());
    }

    public Relu6(INDArray x) {
        super(x);
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[]{cutoff};
    }

    @Override
    public int opNum() {
        return 96;
    }

    @Override
    public String opName() {
        return "relu6";
    }

    @Override
    public String onnxName() { throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "Relu6";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable dLdOut = i_v.get(0);
        return Collections.singletonList(f().relu6Derivative(arg(), dLdOut, cutoff));
    }
}
