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

package org.datavec.api.transform.condition.column;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.SequenceConditionMode;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Set;

/**
 * Condition that applies to the values
 * in a Time column, using a {@link ConditionOp}
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class TimeColumnCondition extends BaseColumnCondition {

    private final ConditionOp op;
    private final Long value;
    private final Set<Long> set;

    /**
     * Constructor for operations such as less than, equal to, greater than, etc.
     * Uses default sequence condition mode, {@link BaseColumnCondition#DEFAULT_SEQUENCE_CONDITION_MODE}
     *
     * @param columnName Column to check for the condition
     * @param op         Operation (<, >=, !=, etc)
     * @param value      Time value (in epoch millisecond format) to use in the condition
     */
    public TimeColumnCondition(String columnName, ConditionOp op, long value) {
        this(columnName, DEFAULT_SEQUENCE_CONDITION_MODE, op, value);
    }

    /**
     * Constructor for operations such as less than, equal to, greater than, etc.
     *
     * @param column                Column to check for the condition
     * @param sequenceConditionMode Mode for handling sequence data
     * @param op                    Operation (<, >=, !=, etc)
     * @param value                 Time value (in epoch millisecond format) to use in the condition
     */
    public TimeColumnCondition(String column, SequenceConditionMode sequenceConditionMode, ConditionOp op, long value) {
        super(column, sequenceConditionMode);
        if (op == ConditionOp.InSet || op == ConditionOp.NotInSet) {
            throw new IllegalArgumentException(
                            "Invalid condition op: cannot use this constructor with InSet or NotInSet ops");
        }
        this.op = op;
        this.value = value;
        this.set = null;
    }

    /**
     * Constructor for operations: ConditionOp.InSet, ConditionOp.NotInSet
     * Uses default sequence condition mode, {@link BaseColumnCondition#DEFAULT_SEQUENCE_CONDITION_MODE}
     *
     * @param column Column to check for the condition
     * @param op     Operation. Must be either ConditionOp.InSet, ConditionOp.NotInSet
     * @param set    Set to use in the condition
     */
    public TimeColumnCondition(String column, ConditionOp op, Set<Long> set) {
        this(column, DEFAULT_SEQUENCE_CONDITION_MODE, op, set);
    }

    /**
     * Constructor for operations: ConditionOp.InSet, ConditionOp.NotInSet
     *
     * @param column                Column to check for the condition
     * @param sequenceConditionMode Mode for handling sequence data
     * @param op                    Operation. Must be either ConditionOp.InSet, ConditionOp.NotInSet
     * @param set                   Set to use in the condition
     */
    public TimeColumnCondition(String column, SequenceConditionMode sequenceConditionMode, ConditionOp op,
                    Set<Long> set) {
        super(column, sequenceConditionMode);
        if (op != ConditionOp.InSet && op != ConditionOp.NotInSet) {
            throw new IllegalArgumentException(
                            "Invalid condition op: can ONLY use this constructor with InSet or NotInSet ops");
        }
        this.op = op;
        this.value = null;
        this.set = set;
    }

    //Private constructor for Jackson deserialization only
    private TimeColumnCondition(@JsonProperty("columnName") String columnName, @JsonProperty("op") ConditionOp op,
                    @JsonProperty("value") long value, @JsonProperty("set") Set<Long> set) {
        super(columnName, DEFAULT_SEQUENCE_CONDITION_MODE);
        this.op = op;
        this.value = (set == null ? value : null);
        this.set = set;
    }


    @Override
    public boolean columnCondition(Writable writable) {
        return op.apply(writable.toLong(), (value == null ? 0 : value), set);
    }

    @Override
    public String toString() {
        return "TimeColumnCondition(columnName=\"" + columnName + "\"," + op + ","
                        + (op == ConditionOp.NotInSet || op == ConditionOp.InSet ? set : value) + ")";
    }

    /**
     * Condition on arbitrary input
     *
     * @param input the input to return
     *              the condition for
     * @return true if the condition is met
     * false otherwise
     */
    @Override
    public boolean condition(Object input) {
        Number n = (Number) input;
        return op.apply(n.longValue(), (value == null ? 0 : value), set);
    }
}
