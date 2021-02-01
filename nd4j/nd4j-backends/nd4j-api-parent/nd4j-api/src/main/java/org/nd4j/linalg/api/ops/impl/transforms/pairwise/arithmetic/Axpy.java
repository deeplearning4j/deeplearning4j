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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;

import java.util.List;

/**
 *  Level 1 blas op Axpy as libnd4j native op
 *
 * @author raver119@gmail.com
 */
public class Axpy extends BaseTransformSameOp {

    private double p;

    public Axpy(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, double p) {
        super(sameDiff, i_v1, i_v2);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, double p) {
        super(sameDiff, i_v1, i_v2, inPlace);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, double p) {
        super(sameDiff);
        this.p = p;
    }

    public Axpy(SameDiff sameDiff, SDVariable i_v, boolean inPlace, double p) {
        super(sameDiff, i_v, inPlace);
        this.p = p;
    }

    public Axpy() {

    }

    public Axpy(INDArray x, INDArray y, INDArray z, double p) {
        this(x,y,z,p,x.length());
    }

    public Axpy(INDArray x, INDArray y, INDArray z, double p, long n) {
        super(x,y,z);
        this.p = p;
        this.extraArgs = new Object[] {p, (double) n};
    }

    @Override
    public int opNum() {
        return 10;
    }

    @Override
    public String opName() {
        return "axpy";
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
        throw new UnsupportedOperationException("Backprop: not yet implemented");
    }
}
