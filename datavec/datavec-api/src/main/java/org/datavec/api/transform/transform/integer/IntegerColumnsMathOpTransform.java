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

package org.datavec.api.transform.transform.integer;

import lombok.Data;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseColumnsMathOpTransform;
import org.datavec.api.transform.transform.doubletransform.DoubleColumnsMathOpTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Add a new integer column, calculated from one or more other columns.
 * A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==MathOp.Add, and columns=={"col1","col2"},
 * then the output column
 * with name "newCol" has value col1+col2.<br>
 * <b>NOTE</b>: Division here is using
 * integer division (integer output). Use {@link DoubleColumnsMathOpTransform}
 * if a decimal output value is required.
 *
 * @author Alex Black
 * @see IntegerMathOpTransform To do an in-place mathematical operation of an integer column and an integer scalar value
 */
@Data
public class IntegerColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    /**
     * @param newColumnName Name of the new column (output column)
     * @param mathOp        Mathematical operation. Only Add/Subtract/Multiply/Divide/Modulus is allowed here
     * @param columns       Columns to use in the mathematical operation
     */
    public IntegerColumnsMathOpTransform(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("mathOp") MathOp mathOp, @JsonProperty("columns") String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData(String newColumnName, Schema inputSchema) {
        return new IntegerMetaData(newColumnName);
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                int sum = 0;
                for (Writable w : input)
                    sum += w.toInt();
                return new IntWritable(sum);
            case Subtract:
                return new IntWritable(input[0].toInt() - input[1].toInt());
            case Multiply:
                int product = 1;
                for (Writable w : input)
                    product *= w.toInt();
                return new IntWritable(product);
            case Divide:
                return new IntWritable(input[0].toInt() / input[1].toInt());
            case Modulus:
                return new IntWritable(input[0].toInt() % input[1].toInt());
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
        return "IntegerColumnsMathOpTransform(newColumnName=\"" + newColumnName + "\",mathOp=" + mathOp + ",columns="
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
        List<Integer> list = (List<Integer>) input;
        switch (mathOp) {
            case Add:
                int sum = 0;
                for (Integer w : list)
                    sum += w;
                return sum;
            case Subtract:
                return new IntWritable(list.get(0) - list.get(1));
            case Multiply:
                int product = 1;
                for (Integer w : list)
                    product *= w;
                return product;
            case Divide:
                return list.get(0) / list.get(1);
            case Modulus:
                return list.get(0) % list.get(1);
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
        List<List<Integer>> seq = (List<List<Integer>>) sequence;
        List<Integer> ret = new ArrayList<>();
        for (List<Integer> step : seq)
            ret.add((Integer) map(step));
        return ret;
    }


}
