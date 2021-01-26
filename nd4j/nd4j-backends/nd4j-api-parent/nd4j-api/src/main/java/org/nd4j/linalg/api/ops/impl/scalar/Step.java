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

package org.nd4j.linalg.api.ops.impl.scalar;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseScalarOp;

import java.util.Collections;
import java.util.List;

/**
 * Unit step function.
 * f(x) = 1 if x > cutoff; 0 otherwise
 * cutoff = 0.0 usually.
 */
public class Step extends BaseScalarOp {
    private final double cutoff;

    public Step(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double cutoff) {
        super(sameDiff, i_v, cutoff, inPlace);
        this.cutoff = cutoff;
        this.extraArgs = new Object[] {cutoff};
    }

    public Step(SameDiff sameDiff, SDVariable i_v, double cutoff) {
        this(sameDiff, i_v, false, cutoff);
    }

    public Step() {
        cutoff = 0.0;
        this.extraArgs = new Object[] {cutoff};
    }

    public Step(INDArray x, INDArray z) {
        super(x, z, 0.0);
        cutoff = 0.0;
        this.extraArgs = new Object[] {cutoff};
    }

    public Step(INDArray x) {
        super(x, 0.0);
        cutoff = 0.0;
        this.extraArgs = new Object[] {cutoff};
    }

    public Step(INDArray x, INDArray z, double cutoff) {
        super(x, z, cutoff);
        this.cutoff = cutoff;
        this.extraArgs = new Object[] {cutoff};
    }

    public Step(INDArray x, double cutoff) {
        super(x, cutoff);
        this.cutoff = cutoff;
        this.extraArgs = new Object[] {cutoff};
    }

    @Override
    public int opNum() {
        return 41;
    }

    @Override
    public String opName() {
        return "step";
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
