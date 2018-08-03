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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;


/**
 * @author Alex Black
 */

public abstract class BaseReduction extends DynamicCustomOp {

    protected boolean keepDims;
    protected int[] dimensions;

    /**
     *
     * @param input        Input to be reduced
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReduction(SameDiff sameDiff, SDVariable input, boolean keepDims, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{input}, false);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param input1   input 1
     * @param input2   input 2
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReduction(SameDiff sameDiff, SDVariable input1, SDVariable input2, boolean keepDims, int... dimensions) {
        super(null, sameDiff, new SDVariable[]{input1, input2}, false);
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param input    input
     * @param output       Output array - i.e., gradient at the input to the reduction function
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReduction(INDArray input, INDArray output, boolean keepDims, int... dimensions){
        super(null, new INDArray[]{input}, (output == null ? null : new INDArray[]{output}));
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    /**
     *
     * @param input1   Pre-reduced input1
     * @param input2   Pre-reduced input2
     * @param output       Output array - i.e., gradient at the input to the reduction function
     * @param keepDims     If true: reduction dimensions were kept
     * @param dimensions   Dimensions to reduce. May be null
     */
    public BaseReduction(INDArray input1, INDArray input2, INDArray output, boolean keepDims, int... dimensions){
        super(null, new INDArray[]{input1, input2}, (output == null ? null : new INDArray[]{output}));
        this.keepDims = keepDims;
        this.dimensions = dimensions;
        addArgs();
    }

    public BaseReduction(){}

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
