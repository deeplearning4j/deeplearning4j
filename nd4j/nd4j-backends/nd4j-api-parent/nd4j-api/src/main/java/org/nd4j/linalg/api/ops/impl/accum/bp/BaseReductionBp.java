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

package org.nd4j.linalg.api.ops.impl.accum.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;


/**
 * @author Alex Black
 */

public abstract class BaseReductionBp extends DynamicCustomOp {

    protected boolean keepDims;
    protected int[] dimensions;

    /**
     *
     * @param origInput    Pre-reduced input
     * @param gradAtOutput Gradient at the output
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReductionBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{origInput, gradAtOutput}, false);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param origInput1   Pre-reduced input 1
     * @param origInput2   Pre-reduced input 2
     * @param gradAtOutput Gradient at the output
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReductionBp(SameDiff sameDiff, SDVariable origInput1, SDVariable origInput2, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{origInput1, origInput2, gradAtOutput}, false);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param origInput    Pre-reduced input
     * @param gradAtOutput Gradient at the output
     * @param output       Output array - i.e., gradient at the input to the reduction function
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReductionBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(null, new INDArray[]{origInput, gradAtOutput}, (output == null ? null : new INDArray[]{output}));
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param origInput1   Pre-reduced input1
     * @param origInput2   Pre-reduced input2
     * @param gradAtOutput Gradient at the output
     * @param output       Output array - i.e., gradient at the input to the reduction function
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReductionBp(INDArray origInput1, INDArray origInput2, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(null, new INDArray[]{origInput1, origInput2, gradAtOutput}, (output == null ? null : new INDArray[]{output}));
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    public BaseReductionBp(){}

    protected void addArgs(){
        addTArgument(keepDims ? 1 : 0);
        if(dimensions != null && dimensions.length > 0){
            if(dimensions.length != 1 || dimensions[0] != Integer.MAX_VALUE ){
                //Integer.MAX_VALUE means "full array" but here no dimension args == full array
                addIArgument(dimensions);
            }
        }
    }

    public abstract String opName();

}
