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

package org.datavec.api.transform.filter;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * A filter based on a {@link Condition}.<br>
 * If condition is satisfied (returns true): remove the example or sequence<br>
 * If condition is not satisfied (returns false): keep the example or sequence
 *
 * @author Alex Black
 */
@EqualsAndHashCode
@Data
public class ConditionFilter implements Filter {

    private final Condition condition;

    public ConditionFilter(@JsonProperty("condition") Condition condition) {
        this.condition = condition;
    }

    /**
     * @param writables Example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeExample(Object writables) {
        return condition.condition(writables);
    }

    /**
     * @param sequence sequence example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeSequence(Object sequence) {
        return condition.condition(sequence);
    }

    @Override
    public boolean removeExample(List<Writable> writables) {
        return condition.condition(writables);
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        return condition.conditionSequence(sequence);
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
        condition.setInputSchema(schema);
    }

    @Override
    public Schema getInputSchema() {
        return condition.getInputSchema();
    }

    @Override
    public String toString() {
        return "ConditionFilter(" + condition + ")";
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return condition.outputColumnName();
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        return condition.outputColumnNames();
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return condition.columnNames();
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return condition.columnName();
    }
}
