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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * DropOut implementation as Op
 *
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public class DropOut extends BaseRandomOp {

    private double p;

    public DropOut(SameDiff sameDiff, SDVariable input, double p) {
        super(sameDiff, input);
        this.p = p;
        //https://github.com/deeplearning4j/deeplearning4j/issues/5650
        throw new UnsupportedOperationException("Dropout SameDiff support disabled pending backprop support");
    }

    public DropOut(@NonNull INDArray x, double p) {
        this(x, x, p, x.lengthLong());
    }

    public DropOut(@NonNull INDArray x, @NonNull INDArray z, double p) {
        this(x, z, p, x.lengthLong());
    }

    public DropOut(@NonNull INDArray x, @NonNull INDArray z, double p, long n) {
        this.p = p;
        init(x, null, z, n);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "dropout";
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {p, (double) n};
    }


    @Override
    public String onnxName() {
        return "Dropout";
    }

    @Override
    public String tensorflowName() {
        return opName();
    }

    @Override
    public Type opType() {
        return Type.RANDOM ;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Not supported");   //We should only use *inverted* dropout with samediff
    }
}
