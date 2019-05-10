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

package org.datavec.api.transform.transform.longtransform;

import lombok.Data;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.LongMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseColumnsMathOpTransform;
import org.datavec.api.transform.transform.doubletransform.DoubleColumnsMathOpTransform;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Add a new long column, calculated from one or more other columns. A new column (with the specified name) is added
 * as the final column of the output. No other columns are modified.<br>
 * For example, if newColumnName=="newCol", mathOp==MathOp.Add, and columns=={"col1","col2"}, then the output column
 * with name "newCol" has value col1+col2.<br>
 * <b>NOTE</b>: Division here is using long division (long output). Use {@link DoubleColumnsMathOpTransform}
 * if a decimal output value is required.
 *
 * @author Alex Black
 * @see LongMathOpTransform To do an in-place mathematical operation of a long column and a long scalar value
 */
@Data
public class LongColumnsMathOpTransform extends BaseColumnsMathOpTransform {

    public LongColumnsMathOpTransform(@JsonProperty("newColumnName") String newColumnName,
                    @JsonProperty("mathOp") MathOp mathOp, @JsonProperty("columns") String... columns) {
        super(newColumnName, mathOp, columns);
    }

    @Override
    protected ColumnMetaData derivedColumnMetaData(String newColumnName, Schema inputSchema) {
        return new LongMetaData(newColumnName);
    }

    @Override
    protected Writable doOp(Writable... input) {
        switch (mathOp) {
            case Add:
                long sum = 0;
                for (Writable w : input)
                    sum += w.toLong();
                return new LongWritable(sum);
            case Subtract:
                return new LongWritable(input[0].toLong() - input[1].toLong());
            case Multiply:
                long product = 1;
                for (Writable w : input)
                    product *= w.toLong();
                return new LongWritable(product);
            case Divide:
                return new LongWritable(input[0].toLong() / input[1].toLong());
            case Modulus:
                return new LongWritable(input[0].toLong() % input[1].toLong());
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
        return "LongColumnsMathOpTransform(newColumnName=\"" + newColumnName + "\",mathOp=" + mathOp + ",columns="
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
        List<Long> list = (List<Long>) input;
        switch (mathOp) {
            case Add:
                long sum = 0;
                for (Long w : list)
                    sum += w;
                return new LongWritable(sum);
            case Subtract:
                return list.get(0) - list.get(1);
            case Multiply:
                long product = 1;
                for (Long w : list)
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
        List<List<Long>> seq = (List<List<Long>>) sequence;
        List<Long> ret = new ArrayList<>();
        for (List<Long> l : seq)
            ret.add((Long) map(l));
        return ret;
    }
}
