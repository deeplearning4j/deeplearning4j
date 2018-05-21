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

package org.datavec.api.transform.condition.column;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.condition.SequenceConditionMode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

/**
 * Abstract class for column conditions
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"columnIdx", "schema", "sequenceMode"})
@EqualsAndHashCode(exclude = {"columnIdx", "schema", "sequenceMode"})
@Data
public abstract class BaseColumnCondition implements ColumnCondition {

    protected final String columnName;
    protected int columnIdx = -1;
    protected Schema schema;
    protected SequenceConditionMode sequenceMode;

    protected BaseColumnCondition(String columnName, SequenceConditionMode sequenceConditionMode) {
        this.columnName = columnName;
        this.sequenceMode = sequenceConditionMode;
    }

    @Override
    public void setInputSchema(Schema schema) {
        columnIdx = schema.getColumnNames().indexOf(columnName);
        if (columnIdx < 0) {
            throw new IllegalStateException("Invalid state: column \"" + columnName + "\" not present in input schema");
        }
        this.schema = schema;
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
    public Schema getInputSchema() {
        return schema;
    }

    @Override
    public boolean condition(List<Writable> list) {
        return columnCondition(list.get(columnIdx));
    }

    @Override
    public boolean conditionSequence(List<List<Writable>> list) {
        switch (sequenceMode) {
            case And:
                for (List<Writable> l : list) {
                    if (!condition(l))
                        return false;
                }
                return true;
            case Or:
                for (List<Writable> l : list) {
                    if (condition(l))
                        return true;
                }
                return false;
            case NoSequenceMode:
                throw new IllegalStateException(
                                "Column condition " + toString() + " does not support sequence execution");
            default:
                throw new RuntimeException("Unknown/not implemented sequence mode: " + sequenceMode);
        }
    }

    @Override
    public boolean conditionSequence(Object list) {
        List<?> objects = (List<?>) list;
        switch (sequenceMode) {
            case And:
                for (Object l : objects) {
                    if (!condition(l))
                        return false;
                }
                return true;
            case Or:
                for (Object l : objects) {
                    if (condition(l))
                        return true;
                }
                return false;
            case NoSequenceMode:
                throw new IllegalStateException(
                                "Column condition " + toString() + " does not support sequence execution");
            default:
                throw new RuntimeException("Unknown/not implemented sequence mode: " + sequenceMode);
        }
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return columnName();
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return columnNames();
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return new String[] {columnName};
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnNames()[0];
    }

    @Override
    public abstract String toString();
}
