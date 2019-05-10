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

package org.datavec.api.transform.transform.floattransform;

import lombok.Data;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.FloatMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.transform.transform.floattransform.FloatColumnsMathOpTransform;
import org.datavec.api.writable.FloatWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Float mathematical operation.<br>
 * This is an in-place operation of the float column value and a float scalar.
 *
 * @author Alex Black
 * @see FloatColumnsMathOpTransform to do a mathematical operation involving multiple columns (instead of a scalar)
 */
@Data
public class FloatMathOpTransform extends BaseColumnTransform {

    private final MathOp mathOp;
    private final float scalar;

    public FloatMathOpTransform(@JsonProperty("columnName") String columnName, @JsonProperty("mathOp") MathOp mathOp,
                                @JsonProperty("scalar") float scalar) {
        super(columnName);
        this.mathOp = mathOp;
        this.scalar = scalar;
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnType) {
        if (!(oldColumnType instanceof FloatMetaData))
            throw new IllegalStateException("Column is not an float column");
        FloatMetaData meta = (FloatMetaData) oldColumnType;
        Float minValue = meta.getMinAllowedValue();
        Float maxValue = meta.getMaxAllowedValue();
        if (minValue != null)
            minValue = doOp(minValue);
        if (maxValue != null)
            maxValue = doOp(maxValue);
        if (minValue != null && maxValue != null && minValue > maxValue) {
            //Consider rsub 1, with original min/max of 0 and 1: (1-0) -> 1 and (1-1) -> 0
            //Or multiplication by -1: (0 to 1) -> (-1 to 0)
            //Need to swap min/max here...
            Float temp = minValue;
            minValue = maxValue;
            maxValue = temp;
        }
        return new FloatMetaData(newColumnName, minValue, maxValue);
    }

    private float doOp(float input) {
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
        return new FloatWritable(doOp(columnWritable.toFloat()));
    }

    @Override
    public String toString() {
        return "FloatMathOpTransform(mathOp=" + mathOp + ",scalar=" + scalar + ")";
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
            return doOp(number.floatValue());
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
