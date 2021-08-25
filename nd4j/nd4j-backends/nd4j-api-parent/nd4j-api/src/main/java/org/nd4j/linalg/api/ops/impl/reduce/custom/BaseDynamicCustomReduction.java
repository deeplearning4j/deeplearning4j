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

import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;

import java.util.Arrays;
import java.util.List;

/**
 * Base class for reduction.
 * 3 main properties matter for any sub class:
 * 1. Dimensions can either be an input variable (SDVariable/INDArray) or int arguments.
 * 2. If you want the dimensions passed as int arguments, pass in the int dimensions array as a constructor.
 * 3. Keep dimensions preserves the rank of the output array relative to the input
 * even when doing a reduce.
 *
 * @author Adam Gibson
 */
public abstract  class BaseDynamicCustomReduction extends DynamicCustomOp {
    @Setter
    @Getter
    protected boolean keepDims = false;
    protected boolean isComplex = false;
    @Setter @Getter
    protected boolean isEmptyReduce = false;
    protected int[] dimensions;

    public BaseDynamicCustomReduction() {}


    public BaseDynamicCustomReduction(SameDiff sameDiff,
                                      SDVariable[] args,
                                      boolean keepDims) {
        this(sameDiff,args,keepDims,false);

    }

    public BaseDynamicCustomReduction(SameDiff sameDiff,
                                      SDVariable[] args,
                                      boolean keepDims,
                                      int[] dimensions) {
        this(sameDiff,args,keepDims,false,dimensions);

    }

    public BaseDynamicCustomReduction(SameDiff sameDiff,
                                      SDVariable[] args,
                                      boolean keepDims,
                                      boolean isComplex) {
        super(null,sameDiff,args);
        this.isComplex = isComplex;
        this.keepDims = keepDims;
        addArgs();

    }

    public BaseDynamicCustomReduction(SameDiff sameDiff,
                                      SDVariable[] args,
                                      boolean keepDims,
                                      boolean isComplex,
                                      int[] dimensions) {
        super(null,sameDiff,args);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.dimensions = dimensions;
        addArgs();


    }

    public BaseDynamicCustomReduction(INDArray[] inputs, INDArray[] outputs) {
        super(null,inputs,outputs);

    }

    public BaseDynamicCustomReduction(INDArray[] inputs, INDArray[] outputs,boolean keepDims) {
        this(inputs,outputs,keepDims,null);
    }

    public BaseDynamicCustomReduction(INDArray[] inputs, INDArray[] outputs,boolean keepDims,int[] dimensions) {
        this(inputs,outputs);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(INDArray[] inputs,boolean keepDims,int[] dimensions) {
        this(inputs,null,keepDims,dimensions);
    }

    public BaseDynamicCustomReduction(boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, arg);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, INDArray input, INDArray output, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, input, output, tArguments, iArguments);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, int[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Integer> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(inputs, outputs);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, inputs, outputs);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName, sameDiff, args, inPlace);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(sameDiff, args, inPlace);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, int[] dimensions) {
        super(opName);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.isEmptyReduce = isEmptyReduce;
        this.dimensions = dimensions;
        addArgs();

    }

    public BaseDynamicCustomReduction(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, int[] dimensions) {
        super(null,input,output);
        this.keepDims = keepDims;
        this.isComplex = isComplex;
        this.dimensions = dimensions;
        addArgs();
    }


    protected void addArgs() {
        addBArgument(keepDims);
        if(dimensions != null) {
            for(int i = 0; i < dimensions.length; i++) {
                addIArgument(dimensions[i]);
            }
        }



    }

    /**
     * Calculate the data types for the output arrays.
     * Though datatypes can also be inferred from {@link #calculateOutputShape()}, this method differs in that it does not
     * require the input arrays to be populated.
     * This is important as it allows us to do greedy datatype inference for the entire net - even if arrays are not
     * available.
     *
     * @param dataTypes The data types of the inputs
     * @return The data types of the outputs
     */
    @Override
    public List<org.nd4j.linalg.api.buffer.DataType> calculateOutputDataTypes(List<org.nd4j.linalg.api.buffer.DataType> dataTypes){
        if(!dArguments.isEmpty())
            return Arrays.asList(dArguments.get(0));
        return Arrays.asList(dataTypes.get(0));
    }



    public abstract String opName();

}
