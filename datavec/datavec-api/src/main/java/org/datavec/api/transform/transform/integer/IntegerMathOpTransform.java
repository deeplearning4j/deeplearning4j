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

package org.datavec.api.transform.transform.integer;

import lombok.Data;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Integer mathematical operation.<br>
 * This is an in-place operation of the integer column value and an integer scalar.
 *
 * @author Alex Black
 * @see IntegerColumnsMathOpTransform to do a mathematical operation involving multiple columns (instead of a scalar)
 */
@Data
public class IntegerMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final int scalar;

    public IntegerMathOpTransform(@JsonProperty("columnName") String columnName, @JsonProperty("mathOp") MathOp mathOp,
                    @JsonProperty("scalar") int scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof IntegerMetaData))
            throw new IllegalStateException("Column is not an integer column");
        IntegerMetaData meta = (IntegerMetaData) oldColumnType;
        Integer minValue = meta.getMinAllowedValue();
        Integer maxValue = meta.getMaxAllowedValue();
        if (minValue != null)
            minValue = doOp(minValue);
        if (maxValue != null)
            maxValue = doOp(maxValue);
        if (minValue != null && maxValue != null && minValue > maxValue) {
            //Consider rsub 1, with original min/max of 0 and 1: (1-0) -> 1 and (1-1) -> 0
            //Or multiplication by -1: (0 to 1) -> (-1 to 0)
            //Need to swap min/max here...
            Integer temp = minValue;
            minValue = maxValue;
            maxValue = temp;
        }
        return new IntegerMetaData(newColumnName, minValue, maxValue);
    }

    private int doOp(int input) {
        switch (mathOp) {
            case Add:
                return input + scalar;
            case Subtract:
                return input - scalar;
            case Multiply:
                return input * scalar;
            case Divide:
                return input / scalar;
            case Modulus:
                return input % scalar;
            case ReverseSubtract:
                return scalar - input;
            case ReverseDivide:
                return scalar / input;
            case ScalarMin:
                return Math.min(input, scalar);
            case ScalarMax:
                return Math.max(input, scalar);
            default:
                throw new IllegalStateException("Unknown or not implemented math op: " + mathOp);
        }
    }

    @Override
    public Writable map(Writable columnWritable) {
        return new IntWritable(doOp(columnWritable.toInt()));
    }

    @Override
    public String toString() {
        return "IntegerMathOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
    }

    /**
     * Transform an object
     * in to another object
     *
     * @param input the record to transform
     * @return the transformed writable
     */
    @Override
    public Object map(Object input) {
        Number n = (Number) input;
        return doOp(n.intValue());
    }


}
