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
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp.RSubBpOp;

import java.util.List;

/**
 * Reverse subtraction operation
 *
 * @author Adam Gibson
 */
public class RSubOp extends BaseDynamicTransformOp {
    public static final String OP_NAME = "reversesubtract";


    public RSubOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace){
        super(sameDiff, args, inPlace);
    }

    public RSubOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        this(sameDiff, new SDVariable[]{i_v1, i_v2}, false);
    }

    public RSubOp(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        this(sameDiff, new SDVariable[]{i_v1, i_v2}, inPlace);
    }

    public RSubOp(INDArray first, INDArray second){
        this(first, second, null);
    }

    public RSubOp(INDArray first, INDArray second, INDArray result){
        this(new INDArray[]{first, second}, wrapOrNull(result));
    }

    public RSubOp( INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public RSubOp() {}

    @Override
    public String opName() {
        return OP_NAME;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No ONNX op name found for: " + getClass().getName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return new RSubBpOp(sameDiff, larg(), rarg(), i_v.get(0)).outputs();
    }

}
