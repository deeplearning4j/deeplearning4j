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

package org.nd4j.linalg.api.ops.impl.transforms.floating;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Collections;
import java.util.List;

/**
 * Gaussian error function (erf) function, which is defined as
 * <p>
 * erf(x) = 1 / sqrt(pi) * integral_(-x, x) exp(-t^2) dt
 *
 * @author raver119@gmail.com
 */
public class Erf extends BaseTransformOp {
    public Erf(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Erf(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public Erf(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public Erf() {
    }

    public Erf(INDArray x, INDArray z) {
        super(x, z);
    }

    public Erf(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public Erf(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public Erf(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 27;
    }

    @Override
    public String opName() {
        return "erf";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Erf";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        // Derivative of erf(z) is 2 / sqrt(pi) * e^(-z^2)
        SDVariable gradient = i_v.get(0);
        SDVariable z = arg();
        SDVariable constant = sameDiff.onesLike(gradient).mul(2.0 / Math.sqrt(Math.PI));
        SDVariable ret = constant.mul(sameDiff.exp(z.mul(z).mul(-1))).mul(gradient);
        return Collections.singletonList(ret);
    }

}
