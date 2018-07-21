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

package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


/**
 * Computes a batch of identity matrices of shape (numRows, numCols), returns a single tensor.
 * This batch of identity matrices can be specified as list of integers.
 *
 * Example:
 *
 * batchShape: [3,3]<br>
 * numRows: 2<br>
 * numCols: 4<br>
 * <br>
 * returns a tensor of shape (3, 3, 2, 4) that consists of 3 * 3 batches of (2,4)-shaped identity matrices:<br>
 * <br>
 *      1 0 0 0<br>
 *      0 1 0 0<br>
 *
 *
 * @author Max Pumperla
 */
public class Eye extends DynamicCustomOp {

    private int numRows;
    private int numCols;
    private boolean isVariableInput = false;
    private int[] batchDimension = new int[] {};

    public Eye() {
    }

    public Eye(SameDiff sameDiff, SDVariable numRows){
        super(null, sameDiff, new SDVariable[] {numRows}, false);
        isVariableInput = true;
    }

    public Eye(SameDiff sameDiff, SDVariable numRows, SDVariable numCols){
        super(null, sameDiff, new SDVariable[] {numRows, numCols}, false);
        isVariableInput = true;
    }
    public Eye(SameDiff sameDiff, SDVariable numRows, SDVariable numCols, SDVariable batch_shape){
        super(null, sameDiff, new SDVariable[] {numRows, numCols, batch_shape}, false);
        isVariableInput = true;
    }
    public Eye(SameDiff sameDiff,  int numRows) {
        super(null, sameDiff, new SDVariable[] {}, false);
        this.numRows = numRows;
        this.numCols = numRows;
        addArgs();
    }

    public Eye(SameDiff sameDiff,  int numRows, int numCols) {
        super(null, sameDiff, new SDVariable[] {}, false);
        this.numRows = numRows;
        this.numCols = numCols;
        addArgs();
    }

    public Eye(SameDiff sameDiff,  int numRows, int numCols, int[] batchDimension) {
        super(null, sameDiff, new SDVariable[] {}, false);
        this.numRows = numRows;
        this.numCols = numCols;
        this.batchDimension = batchDimension;
        addArgs();
    }

    @Override
    public List<long[]> calculateOutputShape(){
        if(isVariableInput){
            return super.calculateOutputShape();
        }
        long[] outputShape = new long[2 + batchDimension.length];
        int i;
        for(i = 0; i < batchDimension.length; i++){
            outputShape[i] = batchDimension[i];
        }
        outputShape[i++] = numRows;
        outputShape[i] = numCols;
        List<long[]> ret = new ArrayList<>();
        ret.add(outputShape);
        return ret;
    }

    protected void addArgs() {
        addIArgument(numRows);
        addIArgument(numCols);
        if(batchDimension != null) {
            for (int dim : batchDimension) {
                addIArgument(dim);
            }
        }
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "Eye";
    }


    @Override
    public String opName() {
        return "eye";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> outGrad){
        return Collections.singletonList(sameDiff.onesLike(arg()));
    }

}
