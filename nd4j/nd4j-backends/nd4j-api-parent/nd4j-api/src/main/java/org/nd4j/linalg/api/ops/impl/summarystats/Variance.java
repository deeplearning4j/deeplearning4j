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

package org.nd4j.linalg.api.ops.impl.summarystats;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.reduce.bp.VarianceBp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Variance extends BaseReduceOp {
    protected double mean, bias;
    protected boolean biasCorrected = true;

    public Variance(SameDiff sameDiff, SDVariable i_v, long[] dimensions, boolean keepDims, double mean) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, boolean keepDims, double mean) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, double mean) {
        super(sameDiff, i_v);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, long[] dimensions, double mean) {
        super(sameDiff, i_v, dimensions);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, double mean) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double mean) {
        super(sameDiff, i_v, keepDims);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double mean) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double mean) {
        super(sameDiff, i_v, i_v2);
        this.mean = mean;
    }

    public Variance(double mean) {
        this.mean = mean;
    }

    public Variance(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions, double mean) {
        super(x, y, z, keepDims, dimensions);
        this.mean = mean;
    }

    public Variance(INDArray x, double mean, long... dimensions) {
        super(x, dimensions);
        this.mean = mean;
    }

    public Variance(INDArray x, boolean keepDims, double mean, long... dimensions) {
        super(x, keepDims, dimensions);
        this.mean = mean;
    }

    public Variance(INDArray x, INDArray y, double mean, long... dimensions) {
        super(x, y, dimensions);
        this.mean = mean;
    }

    public Variance(INDArray x, INDArray y, INDArray z, double mean, long... dimensions) {
        super(x, y, z, dimensions);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, double mean) {
        super(sameDiff);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double mean) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.mean = mean;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, long[] dimensions, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, double mean, double bias) {
        super(sameDiff, i_v);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, long[] dimensions, double mean, double bias) {
        super(sameDiff, i_v, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, double mean, double bias) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, keepDims);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double mean, double bias) {
        super(sameDiff, i_v, i_v2);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(double mean, double bias) {
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions, double mean, double bias) {
        super(x, y, z, keepDims, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(INDArray x, double mean, double bias, long... dimensions) {
        super(x, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(INDArray x, boolean keepDims, double mean, double bias, long... dimensions) {
        super(x, keepDims, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(INDArray x, INDArray y, double mean, double bias, long... dimensions) {
        super(x, y, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(INDArray x, INDArray y, INDArray z, double mean, double bias, long... dimensions) {
        super(x, y, z, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, double mean, double bias) {
        super(sameDiff);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double mean, double bias) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.mean = mean;
        this.bias = bias;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, long[] dimensions, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, long[] dimensions, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, long[] dimensions, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, keepDims);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(double mean, double bias, boolean biasCorrected) {
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, INDArray y, INDArray z, boolean keepDims, long[] dimensions, double mean, double bias, boolean biasCorrected) {
        super(x, y, z, keepDims, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, double mean, double bias, boolean biasCorrected, long... dimensions) {
        super(x, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, boolean keepDims, double mean, double bias, boolean biasCorrected, long... dimensions) {
        super(x, keepDims, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, INDArray y, double mean, double bias, boolean biasCorrected, long... dimensions) {
        super(x, y, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, INDArray y, INDArray z, double mean, double bias, boolean biasCorrected, long... dimensions) {
        super(x, y, z, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, double mean, double bias, boolean biasCorrected) {
        super(sameDiff);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions);
        this.mean = mean;
        this.bias = bias;
        this.biasCorrected = biasCorrected;
    }

    public Variance(SameDiff sameDiff, SDVariable i_v, boolean biasCorrected, boolean keepDims, long[] dimensions) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.biasCorrected = biasCorrected;
        defineDimensions(dimensions);
    }

    public Variance() {
    }

    public Variance(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, long... dimension) {
        this(x, true, dimension);
    }

    public Variance(INDArray x, INDArray z, boolean biasCorrected, long... dimensions) {
        this(x, z, true, false, dimensions);
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, boolean biasCorrected, boolean keepDims, long... dimensions) {
        this(x, null, biasCorrected, keepDims, dimensions);
    }

    public Variance(INDArray x, boolean biasCorrected, long... dimensions) {
        super(x,dimensions);
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, INDArray z, boolean biasCorrected, boolean keepDims, long... dimensions) {
        super(x, null, z, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
    }

    @Override
    public INDArray noOp() {
        return Nd4j.zerosLike(x());
    }

    @Override
    public int opNum() {
        return 0;
    }



    @Override
    public String opName() {
        return "var";
    }


    public boolean isBiasCorrected() {
        return biasCorrected;
    }

    public void setBiasCorrected(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //If out = var(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        // with dOut/dIn = (in-mean) * 2/(n-1)
        return new VarianceBp(sameDiff, arg(), grad.get(0), biasCorrected, keepDims, dimensions).outputs();
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName(){
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public Type getOpType() {
        return Type.VARIANCE;
    }

    @Override
    public DataType resultType() {
        return resultType(null);
    }

    @Override
    public DataType resultType(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        if (x != null && x.isR())
            return x.dataType();

        if(this.arg() != null){
            return this.arg().dataType();
        }

        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public boolean validateDataTypes(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();
        if (x != null && !x.isR()) {
            return false;
        }

        INDArray y = oc != null ? oc.getInputArray(1) : y();
        if (y != null && !y.isR())
            return false;

        INDArray z = oc != null ? oc.getOutputArray(0) : z();
        return !(z != null && !z.isR());
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        return calculateOutputShape(null);
    }

    @Override
    public List<DataBuffer> calculateOutputShape(OpContext oc) {
        INDArray x = oc != null ? oc.getInputArray(0) : x();

        if(oc == null && args().length < 1) {
            throw new ND4JIllegalStateException("Unable to compute input shape. No arguments found.");
        }

        long[] argShape = x.shape();
        if (argShape == null && x == null) {
            return Collections.emptyList();
        }
        long[] inputShape = (argShape == null || Shape.isPlaceholderShape(argShape) ? x.shape() : argShape);

        val ret = new ArrayList<DataBuffer>(1);
        val reducedShape = Shape.getReducedShape(inputShape,dimensions, isKeepDims());
        ret.add(Nd4j.createBuffer(LongShapeDescriptor.fromShape(reducedShape, resultType()).toShapeInfo()));
        return ret;
    }

    @Override
    public Type opType(){
        return Type.VARIANCE;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        //Variance and stdev reduction: Always FP out, but if FP in is float/double/half then it's float/double/half out
        //If not FP in, then return default FP type out
        if(dataTypes.get(0).isFPType())
            return dataTypes;
        return Collections.singletonList(Nd4j.defaultFloatingPointType());
    }
}
