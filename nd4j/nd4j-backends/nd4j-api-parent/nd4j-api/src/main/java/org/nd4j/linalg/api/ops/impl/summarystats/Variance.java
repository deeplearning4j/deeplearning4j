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

package org.nd4j.linalg.api.ops.impl.summarystats;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseReduceOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Variance with bias correction.
 * Bias can either be divided by n or adjusted with:
 * (currentResult - (pow(bias, 2.0) / n())) / (n() - 1.0);
 *
 * @author Adam Gibson
 */
public class Variance extends BaseReduceOp {
    protected double mean, bias;
    protected boolean biasCorrected = true;

    public Variance(SameDiff sameDiff, SDVariable i_v, boolean biasCorrected, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, dimensions, keepDims);
        this.biasCorrected = biasCorrected;
        defineDimensions(dimensions);
    }

    public Variance() {
    }

    public Variance(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, int... dimension) {
        this(x, true, dimension);
    }

    public Variance(INDArray x, INDArray z, boolean biasCorrected, int... dimensions) {
        this(x, z, true, false, dimensions);
        this.biasCorrected = biasCorrected;
    }

    public Variance(INDArray x, boolean biasCorrected, int... dimensions) {
        super(x);
        this.biasCorrected = biasCorrected;
        defineDimensions(dimensions);
    }

    public Variance(INDArray x, INDArray z, boolean biasCorrected, boolean keepDims, int... dimensions) {
        super(x, null, z, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        defineDimensions(dimensions);
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
        return Collections.singletonList(f().varianceBp(arg(), grad.get(0), biasCorrected, keepDims, dimensions));
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
        if (this.x() != null && this.x().isR())
            return this.x().dataType();

        if(this.arg() != null){
            return this.arg().dataType();
        }

        return Nd4j.defaultFloatingPointType();
    }

    @Override
    public boolean validateDataTypes() {
        if (!x().isR())
            return false;

        if (y() != null && !y().isR())
            return false;

        if (z() != null && !z().isR())
            return false;

        return true;
    }

    @Override
    public List<LongShapeDescriptor> calculateOutputShape() {
        if(args().length < 1) {
            throw new ND4JIllegalStateException("Unable to compute input shape. No arguments found.");
        }

        long[] argShape = arg().getShape();
        if (argShape == null && x() == null) {
            return Collections.emptyList();
        }
        long[] inputShape = (argShape == null || Shape.isPlaceholderShape(argShape) ? x().shape() : argShape);

        val ret = new ArrayList<LongShapeDescriptor>(1);
        val reducedShape = Shape.getReducedShape(inputShape,dimensions, isKeepDims());
        ret.add(LongShapeDescriptor.fromShape(reducedShape, resultType()));
        return ret;
    }

    @Override
    public Type opType(){
        return Type.VARIANCE;
    }

    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1, "Expected exactly 1 input datatype for %s, got input %s", getClass(), dataTypes);
        //Variance and stdev reduction: Always FP out, but if FP in is float/double/half then it's float/double/half out
        //If not FP in, then return default FP type out
        if(dataTypes.get(0).isFPType())
            return dataTypes;
        return Collections.singletonList(Nd4j.defaultFloatingPointType());
    }
}
