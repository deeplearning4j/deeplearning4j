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

package org.nd4j.linalg.api.ops.impl.shape.bp;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.*;

/**
 * Backprop op for concat
 *
 * @author Alex Black
 */
@Slf4j
public class ConcatBp extends DynamicCustomOp {
    private int concatDimension;
    private boolean dynamicAxis;

    public ConcatBp(){

    }

    /**
     *
     * @param sameDiff
     * @param concatDimension
     * @param inputsAndGrad     Original inputs, followed by output gradient
     */
    public ConcatBp(@NonNull SameDiff sameDiff, int concatDimension, @NonNull SDVariable... inputsAndGrad){
        super(null, sameDiff, inputsAndGrad);
        addIArgument(concatDimension);
        this.concatDimension = concatDimension;
    }

    /**
     *
     * @param sameDiff       SameDiff instance
     * @param inputsGradAxis Inputs, gradient array, and axis
     */
    public ConcatBp(@NonNull SameDiff sameDiff, @NonNull SDVariable... inputsGradAxis){
        super(null, sameDiff, inputsGradAxis);
        Preconditions.checkState(inputsGradAxis[inputsGradAxis.length-1].dataType().isIntType(),
                "When using this constructor, the last input must be an integer array (for the axis)");
        addBArgument(true);     //Last argument
        this.dynamicAxis = true;
    }

    @Override
    public String opName() {
        return "concat_bp";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public int getNumOutputs(){
        return args().length - 1 - (dynamicAxis ? 1 : 0);
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        SDVariable[] args = args();
        Preconditions.checkState(dataTypes.size() == args.length, "Expected list with exactly %s datatypes (original inputs + gradient), got %s", args.length, dataTypes);
        //Output type is same as (original) input types
        int n = getNumOutputs();
        List<DataType> out = new ArrayList<>(n);
        for( int i=0; i<n; i++){
            out.add(arg(i).dataType());
        }
        return out;
    }
}
