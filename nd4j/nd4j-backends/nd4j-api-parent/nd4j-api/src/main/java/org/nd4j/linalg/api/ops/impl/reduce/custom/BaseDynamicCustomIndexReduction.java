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
package org.nd4j.linalg.api.ops.impl.reduce.custom;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;

public abstract class BaseDynamicCustomIndexReduction extends BaseDynamicCustomReduction {

    public BaseDynamicCustomIndexReduction() {
        super();
    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable[] args, boolean keepDims, long[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
        addDArgument(DataType.INT64);
    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, long[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }

    public BaseDynamicCustomIndexReduction(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public BaseDynamicCustomIndexReduction(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
        addDArgument(DataType.INT64);
    }


    public BaseDynamicCustomIndexReduction(INDArray[] inputs, INDArray[] outputs, boolean keepDims,long...dimensions) {
        super(inputs, outputs, keepDims,dimensions);
        addDArgument(DataType.INT64);
    }

    public BaseDynamicCustomIndexReduction(INDArray[] inputs, boolean keepDims, long[] dimensions) {
        super(inputs, keepDims, dimensions);
        addDArgument(DataType.INT64);


    }

    public BaseDynamicCustomIndexReduction(boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, INDArray input, INDArray output, List<Double> tArguments, long[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, long[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Long> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
        addDArgument(DataType.INT64);

    }

    public BaseDynamicCustomIndexReduction(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, long[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
        addDArgument(DataType.INT64);

    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        return calculateOutputShape(null);
    }



    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //All reduce long ops: always long output type
        //Second input is dynamic axis arg
        Preconditions.checkState(dataTypes != null && (dataTypes.size() == 1 || dataTypes.size() == 2),
                "Expected 1 or input datatype for %s, got input %s", getClass(), dataTypes);
        Preconditions.checkState(dataTypes.size() == 1 || dataTypes.get(1).isIntType(), "When executing reductions" +
                "with 2 inputs, second input (axis) must be an integer datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(DataType.INT64);
    }

}
