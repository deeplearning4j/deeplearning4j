/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.api.transform.ndarray;

import lombok.Data;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseColumnsMathOpTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;

/**
 * Perform an element wise mathematical operation on 2 or more NDArray columns
 *
 * @author Alex Black
 */
@Data
public class NDArrayColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    public NDArrayColumnsMathOpTransform(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("mathOp") MathOp mathOp, @JsonProperty("columns") String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData(String newColumnName, Schema inputSchema) {
        //Check types

        for (int i = 0; i < columns.length; i++) {
            if (inputSchema.getMetaData(columns[i]).getColumnType() != ColumnType.NDArray) {
                throw new RuntimeException("Column " + columns[i] + " is not an NDArray column");
            }
        }

        //Check shapes
        NDArrayMetaData meta = (NDArrayMetaData) inputSchema.getMetaData(columns[0]);
        for (int i = 1; i < columns.length; i++) {
            NDArrayMetaData meta2 = (NDArrayMetaData) inputSchema.getMetaData(columns[i]);
            if (!Arrays.equals(meta.getShape(), meta2.getShape())) {
                throw new UnsupportedOperationException(
                                "Cannot perform NDArray operation on columns with different shapes: " + "Columns \""
                                                + columns[0] + "\" and \"" + columns[i] + "\" have shapes: "
                                                + Arrays.toString(meta.getShape()) + " and "
                                                + Arrays.toString(meta2.getShape()));
            }
        }

        return new NDArrayMetaData(newColumnName, meta.getShape());
    }

    @Override
    protected Writable doOp(Writable... input) {
        INDArray out = ((NDArrayWritable) input[0]).get().dup();

        switch (mathOp) {
            case Add:
                for (int i = 1; i < input.length; i++) {
                    out.addi(((NDArrayWritable) input[i]).get());
                }
                break;
            case Subtract:
                out.subi(((NDArrayWritable) input[1]).get());
                break;
            case Multiply:
                for (int i = 1; i < input.length; i++) {
                    out.muli(((NDArrayWritable) input[i]).get());
                }
                break;
            case Divide:
                out.divi(((NDArrayWritable) input[1]).get());
                break;
            case ReverseSubtract:
                out.rsubi(((NDArrayWritable) input[1]).get());
                break;
            case ReverseDivide:
                out.rdivi(((NDArrayWritable) input[1]).get());
                break;
            case Modulus:
            case ScalarMin:
            case ScalarMax:
                throw new IllegalArgumentException(
                                "Invalid MathOp: cannot use " + mathOp + " with NDArrayColumnsMathOpTransform");
            default:
                throw new RuntimeException("Unknown MathOp: " + mathOp);
        }

        //To avoid threading issues...
        Nd4j.getExecutioner().commit();

        return new NDArrayWritable(out);
    }

    @Override
    public String toString() {
        return "NDArrayColumnsMathOpTransform(newColumnName=\"" + newColumnName + "\",mathOp=" + mathOp + ",columns="
                        + Arrays.toString(columns) + ")";
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
