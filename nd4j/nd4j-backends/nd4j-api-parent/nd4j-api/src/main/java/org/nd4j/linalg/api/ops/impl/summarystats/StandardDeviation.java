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


import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.bp.StandardDeviationBp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class StandardDeviation extends Variance {
    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, boolean biasCorrected, boolean keepDims, int[] dimensions) {
        super(sameDiff, i_v, biasCorrected, keepDims, dimensions);
        this.keepDims = keepDims;
    }

    public StandardDeviation(INDArray x, boolean biasCorrected, boolean keepDims, int... dimension) {
        super(x, biasCorrected, dimension);
        this.keepDims = keepDims;
    }

    public StandardDeviation(INDArray x, boolean biasCorrected, int... dimension) {
        super(x, biasCorrected, dimension);
    }


    public StandardDeviation() {
    }

    public StandardDeviation(boolean biasCorrected) {
        super(biasCorrected);
    }

    public StandardDeviation(INDArray x, int... dimension) {
        super(x, dimension);
    }

    public StandardDeviation(INDArray x) {
        super(x);
    }

    public StandardDeviation(INDArray x, INDArray z, boolean biasCorrected, int... dimension) {
        super(x, z, biasCorrected, dimension);
    }

    public StandardDeviation(INDArray x, INDArray z, boolean newFormat, boolean keepDims, int[] dimensions) {
        super(x, z, newFormat, keepDims, dimensions);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, boolean keepDims, double mean) {
        super(sameDiff, i_v, dimensions, keepDims, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims, double mean) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, double mean) {
        super(sameDiff, i_v, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, double mean) {
        super(sameDiff, i_v, dimensions, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double mean) {
        super(sameDiff, i_v, i_v2, dimensions, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double mean) {
        super(sameDiff, i_v, keepDims, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double mean) {
        super(sameDiff, i_v, dimensions, keepDims, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double mean) {
        super(sameDiff, i_v, i_v2, mean);
    }

    public StandardDeviation(double mean) {
        super(mean);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions, double mean) {
        super(x, y, z, keepDims, dimensions, mean);
    }

    public StandardDeviation(INDArray x, double mean, int... dimensions) {
        super(x, mean, dimensions);
    }

    public StandardDeviation(INDArray x, boolean keepDims, double mean, int... dimensions) {
        super(x, keepDims, mean, dimensions);
    }

    public StandardDeviation(INDArray x, INDArray y, double mean, int... dimensions) {
        super(x, y, mean, dimensions);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, double mean, int... dimensions) {
        super(x, y, z, mean, dimensions);
    }

    public StandardDeviation(SameDiff sameDiff, double mean) {
        super(sameDiff, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double mean) {
        super(sameDiff, i_v, i_v2, dimensions, mean);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, dimensions, keepDims, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, double mean, double bias) {
        super(sameDiff, i_v, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, double mean, double bias) {
        super(sameDiff, i_v, dimensions, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double mean, double bias) {
        super(sameDiff, i_v, i_v2, dimensions, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, keepDims, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double mean, double bias) {
        super(sameDiff, i_v, dimensions, keepDims, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double mean, double bias) {
        super(sameDiff, i_v, i_v2, mean, bias);
    }

    public StandardDeviation(double mean, double bias) {
        super(mean, bias);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions, double mean, double bias) {
        super(x, y, z, keepDims, dimensions, mean, bias);
    }

    public StandardDeviation(INDArray x, double mean, double bias, int... dimensions) {
        super(x, mean, bias, dimensions);
    }

    public StandardDeviation(INDArray x, boolean keepDims, double mean, double bias, int... dimensions) {
        super(x, keepDims, mean, bias, dimensions);
    }

    public StandardDeviation(INDArray x, INDArray y, double mean, double bias, int... dimensions) {
        super(x, y, mean, bias, dimensions);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, double mean, double bias, int... dimensions) {
        super(x, y, z, mean, bias, dimensions);
    }

    public StandardDeviation(SameDiff sameDiff, double mean, double bias) {
        super(sameDiff, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double mean, double bias) {
        super(sameDiff, i_v, i_v2, dimensions, mean, bias);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, keepDims, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions, keepDims, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, int[] dimensions, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, int[] dimensions, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, keepDims, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable dimensions, boolean keepDims, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, dimensions, keepDims, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, mean, bias, biasCorrected);
    }

    public StandardDeviation(double mean, double bias, boolean biasCorrected) {
        super(mean, bias, biasCorrected);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, boolean keepDims, int[] dimensions, double mean, double bias, boolean biasCorrected) {
        super(x, y, z, keepDims, dimensions, mean, bias, biasCorrected);
    }

    public StandardDeviation(INDArray x, double mean, double bias, boolean biasCorrected, int... dimensions) {
        super(x, mean, bias, biasCorrected, dimensions);
    }

    public StandardDeviation(INDArray x, boolean keepDims, double mean, double bias, boolean biasCorrected, int... dimensions) {
        super(x, keepDims, mean, bias, biasCorrected, dimensions);
    }

    public StandardDeviation(INDArray x, INDArray y, double mean, double bias, boolean biasCorrected, int... dimensions) {
        super(x, y, mean, bias, biasCorrected, dimensions);
    }

    public StandardDeviation(INDArray x, INDArray y, INDArray z, double mean, double bias, boolean biasCorrected, int... dimensions) {
        super(x, y, z, mean, bias, biasCorrected, dimensions);
    }

    public StandardDeviation(SameDiff sameDiff, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, mean, bias, biasCorrected);
    }

    public StandardDeviation(SameDiff sameDiff, SDVariable i_v, SDVariable i_v2, SDVariable dimensions, double mean, double bias, boolean biasCorrected) {
        super(sameDiff, i_v, i_v2, dimensions, mean, bias, biasCorrected);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "std";
    }

    @Override
    public String onnxName(){
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName(){
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public Type getOpType() {
        return Type.SUMMARYSTATS;
    }

    @Override
    public Type opType(){
        return Type.SUMMARYSTATS;
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        Boolean isEmptyReduce = getBooleanFromProperty("isEmptyReduce",properties);
        if(isEmptyReduce != null) {
            this.isEmptyReduce = isEmptyReduce;
        }

        Boolean biasCorrected = getBooleanFromProperty("biasCorrected",properties);
        if(biasCorrected != null) {
            this.biasCorrected = biasCorrected;
        }

        Double mean = getDoubleValueFromProperty("mean",properties);
        if(mean != null) {
            this.mean = mean;
        }

        Boolean keepDims = getBooleanFromProperty("keepDims",properties);
        if(keepDims != null) {
            this.keepDims = keepDims;
        }

        Boolean isComplex = getBooleanFromProperty("isComplex",properties);
        if(isComplex != null) {
            this.isComplex = isComplex;
        }

        Double bias = getDoubleValueFromProperty("bias",properties);
        if(bias != null) {
            this.bias = bias;
        }



    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return super.calculateOutputDataTypes(dataTypes);
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad) {
        //Here: calculating dL/dIn given dL/dOut (i.e., i_v1) and input/output
        //If out = stdev(in) then:
        //dL/dIn = dL/dOut * dOut/dIn
        //dOut/dIn_i = (in_i-mean)/(stdev * (n-1))
        return new StandardDeviationBp(sameDiff, arg(), grad.get(0), biasCorrected, keepDims, dimensions).outputs();
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

        var ret = new ArrayList<LongShapeDescriptor>(1);
        var reducedShape = Shape.getReducedShape(inputShape,dimensions, isKeepDims());
        ret.add(LongShapeDescriptor.fromShape(reducedShape, resultType()));
        return ret;
    }
}
