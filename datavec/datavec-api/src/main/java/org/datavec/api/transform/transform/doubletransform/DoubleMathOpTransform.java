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

package org.datavec.api.transform.transform.doubletransform;

import lombok.Data;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Double mathematical operation.<br>
 * This is an in-place operation of the double column value and a double scalar.
 *
 * @author Alex Black
 * @see DoubleColumnsMathOpTransform to do a mathematical operation involving multiple columns (instead of a scalar)
 */
@Data
public class DoubleMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final double scalar;

    public DoubleMathOpTransform(@JsonProperty("columnName") String columnName, @JsonProperty("mathOp") MathOp mathOp,
                    @JsonProperty("scalar") double scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof DoubleMetaData))
            throw new IllegalStateException("Column is not an double column");
        DoubleMetaData meta = (DoubleMetaData) oldColumnType;
        Double minValue = meta.getMinAllowedValue();
        Double maxValue = meta.getMaxAllowedValue();
        if (minValue != null)
            minValue = doOp(minValue);
        if (maxValue != null)
            maxValue = doOp(maxValue);
        if (minValue != null && maxValue != null && minValue > maxValue) {
            //Consider rsub 1, with original min/max of 0 and 1: (1-0) -> 1 and (1-1) -> 0
            //Or multiplication by -1: (0 to 1) -> (-1 to 0)
            //Need to swap min/max here...
            Double temp = minValue;
            minValue = maxValue;
            maxValue = temp;
        }
        return new DoubleMetaData(newColumnName, minValue, maxValue);
    }

    private double doOp(double input) {
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
        return new DoubleWritable(doOp(columnWritable.toDouble()));
    }

    @Override
    public String toString() {
        return "DoubleMathOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
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
        if (input instanceof Number) {
            Number number = (Number) input;
            return doOp(number.doubleValue());
        }
        throw new IllegalArgumentException("Input must be a number");
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<?> list = (List<?>) sequence;
        List<Object> ret = new ArrayList<>();
        for (Object o : list)
            ret.add(map(o));
        return ret;
    }
}
