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

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MaxOut extends BaseTransformOp {

    private Number max = Double.NaN;

    public MaxOut(SameDiff sameDiff, SDVariable i_v, boolean inPlace, Number max) {
        super(sameDiff, i_v, inPlace);
        this.max = max;
    }

    public MaxOut(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs, Number max) {
        super(sameDiff, i_v, extraArgs);
        this.max = max;
    }

    public MaxOut() {}

    public MaxOut(INDArray x, INDArray z) {
        super(x, z);
    }

    public MaxOut(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        throw new UnsupportedOperationException();
    }


    @Override
    public String opName() {
        return "maxout";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }


    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("Tensorflow name not found for " + opName());
        //return "Maxout";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

    @Override
    public DataType resultType() {
        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public DataType resultType(OpContext oc) {
        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public Type getOpType() {
        return Type.TRANSFORM_STRICT;
    }

    @Override
    public boolean validateDataTypes(OpContext oc, boolean experimentalMode) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        INDArray y = oc != null ? oc.getInputArray(1) : y();
        INDArray z = oc != null ? oc.getOutputArray(0) : z();

        if (!x.isR())
            return false;

        if (y != null && !y().isR())
            return false;

        if (z != null && z().dataType() != x().dataType())
            return false;

        return true;
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        val ret = new ArrayList<DataBuffer>(1);
        if(arg() == null)
            throw new ND4JIllegalStateException("No arg found for op!");

        val arr = sameDiff.getArrForVarName(arg().name());
        if(arr == null)
            return Collections.emptyList();

        ret.add(Nd4j.createBuffer(LongShapeDescriptor.fromShape(arr.shape(), Nd4j.defaultFloatingPointType()).toShapeInfo()));
        return ret;
    }
}
