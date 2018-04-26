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
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseColumnsMathOpTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Add a new double column, calculated from one or more other columns. A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==Add, and columns=={"col1","col2"}, then the output column
 * with name "newCol" has value col1+col2.<br>
 *
 * @author Alex Black
 * @see DoubleMathOpTransform To do an in-place mathematical operation of a double column and a double scalar value
 */
@Data
public class DoubleColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    public DoubleColumnsMathOpTransform(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("mathOp") MathOp mathOp, @JsonProperty("columns") List<String> columns) {
        this(newColumnName, mathOp, columns.toArray(new String[columns.size()]));
    }

    public DoubleColumnsMathOpTransform(String newColumnName, MathOp mathOp, String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData(String newColumnName, Schema inputSchema) {
        return new DoubleMetaData(newColumnName);
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                double sum = 0;
                for (Writable w : input)
                    sum += w.toDouble();
                return new DoubleWritable(sum);
            case Subtract:
                return new DoubleWritable(input[0].toDouble() - input[1].toDouble());
            case Multiply:
                double product = 1.0;
                for (Writable w : input)
                    product *= w.toDouble();
                return new DoubleWritable(product);
            case Divide:
                return new DoubleWritable(input[0].toDouble() / input[1].toDouble());
            case Modulus:
                return new DoubleWritable(input[0].toDouble() % input[1].toDouble());
            case ReverseSubtract:
            case ReverseDivide:
            case ScalarMin:
            case ScalarMax:
            default:
                throw new RuntimeException("Invalid mathOp: " + mathOp); //Should never happen
        }
    }

    @Override
    public String toString() {
        return "DoubleColumnsMathOpTransform(newColumnName=\"" + newColumnName + "\",mathOp=" + mathOp + ",columns="
                        + Arrays.toString(columns) + ")";
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
        List<Double> row = (List<Double>) input;
        switch (mathOp) {
            case Add:
                double sum = 0;
                for (Double w : row)
                    sum += w;
                return sum;
            case Subtract:
                return row.get(0) - row.get(1);
            case Multiply:
                double product = 1.0;
                for (Double w : row)
                    product *= w;
                return product;
            case Divide:
                return row.get(0) / row.get(1);
            case Modulus:
                return row.get(0) % row.get(1);
            case ReverseSubtract:
            case ReverseDivide:
            case ScalarMin:
            case ScalarMax:
            default:
                throw new RuntimeException("Invalid mathOp: " + mathOp); //Should never happen
        }
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<List<Double>> seq = (List<List<Double>>) sequence;
        List<Double> ret = new ArrayList<>();
        for (List<Double> step : seq)
            ret.add((Double) map(step));
        return ret;
    }
}
