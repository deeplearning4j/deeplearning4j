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

package org.datavec.api.transform.condition;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * BooleanCondition: used for creating compound conditions,
 * such as AND(ConditionA, ConditionB, ...)<br>
 * As a BooleanCondition is a condition,
 * these can be chained together,
 * like NOT(OR(AND(...),AND(...)))
 *
 * @author Alex Black
 */
@EqualsAndHashCode
@Data
public class BooleanCondition implements Condition {

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return conditions[0].outputColumnName();
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return conditions[0].outputColumnNames();
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return conditions[0].columnNames();
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return conditions[0].columnName();
    }

    public enum Type {
        AND, OR, NOT, XOR
    }

    private final Type type;
    private final Condition[] conditions;

    public BooleanCondition(@JsonProperty("type") Type type, @JsonProperty("conditions") Condition... conditions) {
        if (conditions == null || conditions.length < 1)
            throw new IllegalArgumentException(
                            "Invalid input: conditions must be non-null and have at least 1 element");
        switch (type) {
            case NOT:
                if (conditions.length != 1)
                    throw new IllegalArgumentException("Invalid input: NOT conditions must have exactly 1 element");
                break;
            case XOR:
                if (conditions.length != 2)
                    throw new IllegalArgumentException("Invalid input: XOR conditions must have exactly 2 elements");
                break;
        }
        this.type = type;
        this.conditions = conditions;
    }

    @Override
    public boolean condition(List<Writable> list) {
        switch (type) {
            case AND:
                for (Condition c : conditions) {
                    boolean thisCond = c.condition(list);
                    if (!thisCond)
                        return false; //Any false -> AND is false
                }
                return true;
            case OR:
                for (Condition c : conditions) {
                    boolean thisCond = c.condition(list);
                    if (thisCond)
                        return true; //Any true -> OR is true
                }
                return false;
            case NOT:
                return !conditions[0].condition(list);
            case XOR:
                return conditions[0].condition(list) ^ conditions[1].condition(list);
            default:
                throw new RuntimeException("Unknown condition type: " + type);
        }
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
        switch (type) {
            case AND:
                for (Condition c : conditions) {
                    boolean thisCond = c.condition(input);
                    if (!thisCond)
                        return false; //Any false -> AND is false
                }
                return true;
            case OR:
                for (Condition c : conditions) {
                    boolean thisCond = c.condition(input);
                    if (thisCond)
                        return true; //Any true -> OR is true
                }
                return false;
            case NOT:
                return !conditions[0].condition(input);
            case XOR:
                return conditions[0].condition(input) ^ conditions[1].condition(input);
            default:
                throw new RuntimeException("Unknown condition type: " + type);
        }
    }

    @Override
    public boolean conditionSequence(List<List<Writable>> sequence) {
        switch (type) {
            case AND:
                for (Condition c : conditions) {
                    boolean thisCond = c.conditionSequence(sequence);
                    if (!thisCond)
                        return false; //Any false -> AND is false
                }
                return true;
            case OR:
                for (Condition c : conditions) {
                    boolean thisCond = c.conditionSequence(sequence);
                    if (thisCond)
                        return true; //Any true -> OR is true
                }
                return false;
            case NOT:
                return !conditions[0].conditionSequence(sequence);
            case XOR:
                return conditions[0].conditionSequence(sequence) ^ conditions[1].conditionSequence(sequence);
            default:
                throw new RuntimeException("Unknown condition type: " + type);
        }
    }

    /**
     * Condition on arbitrary input
     *
     * @param sequence the sequence to
     *                 do a condition on
     * @return true if the condition for the sequence is met false otherwise
     */
    @Override
    public boolean conditionSequence(Object sequence) {
        List<?> seq = (List<?>) sequence;
        switch (type) {
            case AND:
                for (Condition c : conditions) {
                    boolean thisCond = c.conditionSequence(seq);
                    if (!thisCond)
                        return false; //Any false -> AND is false
                }
                return true;
            case OR:
                for (Condition c : conditions) {
                    boolean thisCond = c.conditionSequence(seq);
                    if (thisCond)
                        return true; //Any true -> OR is true
                }
                return false;
            case NOT:
                return !conditions[0].conditionSequence(sequence);
            case XOR:
                return conditions[0].conditionSequence(sequence) ^ conditions[1].conditionSequence(seq);
            default:
                throw new RuntimeException("Unknown condition type: " + type);
        }
    }

    /**
     * Get the output schema for this transformation, given an input schema
     *
     * @param inputSchema
     */
    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema;
    }

    @Override
    public void setInputSchema(Schema schema) {
        for (Condition c : conditions) {
            c.setInputSchema(schema);
        }
    }

    @Override
    public Schema getInputSchema() {
        return conditions[0].getInputSchema();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("BooleanCondition(").append(type);
        for (Condition c : conditions) {
            sb.append(",").append(c.toString());
        }
        sb.append(")");
        return sb.toString();
    }


    /**
     * And of all the given conditions
     * @param conditions the conditions to and
     * @return a joint and of all these conditions
     */
    public static Condition AND(Condition... conditions) {
        return new BooleanCondition(Type.AND, conditions);
    }

    /**
     * Or of all the given conditions
     * @param conditions the conditions to or
     * @return a joint and of all these conditions
     */
    public static Condition OR(Condition... conditions) {
        return new BooleanCondition(Type.OR, conditions);
    }

    /**
     * Not of  the given condition
     * @param condition the conditions to and
     * @return a joint and of all these condition
     */
    public static Condition NOT(Condition condition) {
        return new BooleanCondition(Type.NOT, condition);
    }

    /**
     * And of all the given conditions
     * @param first the first condition
     * @param second  the second condition for xor
     * @return the xor of these 2 conditions
     */
    public static Condition XOR(Condition first, Condition second) {
        return new BooleanCondition(Type.XOR, first, second);
    }


}
