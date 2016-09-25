/*
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

import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.transform.doubletransform.DoubleColumnsMathOpTransform;
import org.datavec.api.transform.transform.BaseColumnsMathOpTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;

import java.util.Arrays;

/**
 * Add a new integer column, calculated from one or more other columns. A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==MathOp.Add, and columns=={"col1","col2"}, then the output column
 * with name "newCol" has value col1+col2.<br>
 * <b>NOTE</b>: Division here is using integer division (integer output). Use {@link DoubleColumnsMathOpTransform}
 * if a decimal output value is required.
 *
 * @author Alex Black
 * @see IntegerMathOpTransform To do an in-place mathematical operation of an integer column and an integer scalar value
 */
public class IntegerColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    /**
     * @param newColumnName Name of the new column (output column)
     * @param mathOp        Mathematical operation. Only Add/Subtract/Multiply/Divide/Modulus is allowed here
     * @param columns       Columns to use in the mathematical operation
     */
    public IntegerColumnsMathOpTransform(@JsonProperty("newColumnName") String newColumnName, @JsonProperty("mathOp") MathOp mathOp,
                                         @JsonProperty("columns") String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData(String newColumnName) {
        return new IntegerMetaData(newColumnName);
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                int sum = 0;
                for (Writable w : input) sum += w.toInt();
                return new IntWritable(sum);
            case Subtract:
                return new IntWritable(input[0].toInt() - input[1].toInt());
            case Multiply:
                int product = 1;
                for (Writable w : input) product *= w.toInt();
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
                throw new RuntimeException("Invalid mathOp: " + mathOp);    //Should never happen
        }
    }

    @Override
    public String toString(){
        return "IntegerColumnsMathOpTransform(newColumnName=\"" + newColumnName + "\",mathOp=" + mathOp + ",columns=" + Arrays.toString(columns) + ")";
    }
}
