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

package org.nd4j.linalg.api.ops.impl.accum.distances;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseAccumulation;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * Hamming distance (simple)
 *
 * @author raver119@gmail.com
 */
public class HammingDistance extends BaseAccumulation {


    public HammingDistance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int... dimensions) {
        super(sameDiff, i_v, i_v2, dimensions);
    }

    public HammingDistance() {
        passThrough = true;
    }

    public HammingDistance(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public HammingDistance(INDArray x, INDArray y, long n) {
        super(x, y, n);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public HammingDistance(INDArray x) {
        super(x);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public HammingDistance(INDArray x, INDArray y) {
        super(x, y);
        passThrough = Nd4j.getExecutioner().executionMode() == OpExecutioner.ExecutionMode.JAVA;
        extraArgs = new Object[2];
        extraArgs[0] = 0.0f;
        extraArgs[1] = 0.0f;
    }

    public HammingDistance(INDArray x, INDArray y, INDArray z, boolean allDistances) {
        this(x, y, z, x.lengthLong());
        this.isComplex = allDistances;
    }

    public HammingDistance(INDArray x, INDArray y, boolean allDistances) {
        this(x, y);
        this.isComplex = allDistances;
    }

    @Override
    public Type opType() {
        return Type.REDUCE3;
    }

    @Override
    public Type getOpType() {
        return opType();
    }

    @Override
    public int opNum() {
        return 7;
    }

    @Override
    public String opName() {
        return "hammingdistance";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //Hamming distance: "the Hamming distance between two strings of equal length is the number of positions at
        // which the corresponding symbols are different."
        //Consequently: it's not continuously differentiable, and gradients are 0 almost everywhere (but undefined
        // when x_i == y_i)
        return Arrays.asList(sameDiff.zerosLike(larg()), sameDiff.zerosLike(rarg()));
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
