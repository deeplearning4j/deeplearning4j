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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.shade.guava.base.Preconditions;

import java.util.Collections;
import java.util.List;


public class Eye extends DynamicCustomOp {
    public static final DataType DEFAULT_DTYPE = DataType.FLOAT;

    private int numRows;
    private int numCols;
    private long[] batchDimension = new long[] {};
    private DataType dataType = DEFAULT_DTYPE;

    public Eye() {
    }

    public Eye(@NonNull INDArray rows){
        this(rows.getInt(0));
        Preconditions.checkArgument(rows.isScalar(), "Rows INDArray must be a scalar");
    }

    public Eye(@NonNull INDArray rows, @NonNull INDArray columns){
        this(rows.getInt(0), columns.getInt(0));
        Preconditions.checkArgument(rows.isScalar(), "Rows INDArray must be a scalar");
        Preconditions.checkArgument(columns.isScalar(), "Columns INDArray must be a scalar");
    }

    public Eye(int rows){
        this.numRows = rows;
        this.numCols = rows;
        addArgs();
    }

    public Eye(SameDiff sameDiff, SDVariable numRows){
        super(null, sameDiff, new SDVariable[] {numRows}, false);
    }

    public Eye(SameDiff sameDiff, SDVariable numRows, SDVariable numCols){
        super(null, sameDiff, new SDVariable[] {numRows, numCols}, false);
    }
    public Eye(SameDiff sameDiff, SDVariable numRows, SDVariable numCols, SDVariable batch_shape){
        super(null, sameDiff, new SDVariable[] {numRows, numCols, batch_shape}, false);
    }
    public Eye(SameDiff sameDiff,  int numRows) {
        this(sameDiff, numRows, numRows);
    }

    public Eye(SameDiff sameDiff,  int numRows, int numCols) {
        this(sameDiff, numRows, numCols, DEFAULT_DTYPE);
    }

    public Eye(SameDiff sameDiff,  int numRows, int numCols, DataType dataType) {
        this(sameDiff, numRows, numCols, dataType, null);
    }

    public Eye(int numRows, int numCols, DataType dataType, long[] batchDimension) {
        this.numRows = numRows;
        this.numCols = numCols;
        this.batchDimension = batchDimension;
        this.dataType = dataType;
        addArgs();
    }

    public Eye(int numRows, int numCols) {
        this(numRows, numCols, DEFAULT_DTYPE);
    }

    public Eye(int numRows, int numCols, DataType dataType) {
        this(numRows, numCols, dataType, null);
    }



    public Eye(SameDiff sameDiff, int numRows, int numCols, DataType dataType, long[] batchDimension) {
        super(null, sameDiff, new SDVariable[] {});
        this.batchDimension = batchDimension;
        this.dataType = dataType;
        this.numRows = numRows;
        this.numCols = numCols;
        addArgs();
    }

    protected void addArgs() {
        iArguments.clear();
        tArguments.clear();

        addIArgument(numRows);
        addIArgument(numCols);
        if(batchDimension != null) {
            for (long dim : batchDimension) {
                addIArgument(dim);
            }
        }

        addTArgument((double) dataType.toInt());
    }

    @Override
    public String opName() {
        return "eye";
    }

    @Override
    public List<DataBuffer> calculateOutputShape() {
        List<DataBuffer> l = super.calculateOutputShape();
        return l;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> outGrad){
        if(arg() != null){
            return Collections.singletonList(sameDiff.onesLike(arg()));
        } else {
            return Collections.emptyList();
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        return Collections.singletonList(dataType == null ? DEFAULT_DTYPE : dataType);
    }

}
