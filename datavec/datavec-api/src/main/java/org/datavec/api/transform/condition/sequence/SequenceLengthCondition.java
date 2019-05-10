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

package org.datavec.api.transform.condition.sequence;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Set;

/**
 * A condition on sequence lengths
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema"})
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
public class SequenceLengthCondition implements Condition {

    private ConditionOp op;
    private Integer length;
    private Set<Integer> set;

    private Schema inputSchema;

    public SequenceLengthCondition(ConditionOp op, int length) {
        this(op, length, null);
    }

    public SequenceLengthCondition(ConditionOp op, Set<Integer> set) {
        this(op, null, set);
    }

    private SequenceLengthCondition(@JsonProperty("op") ConditionOp op, @JsonProperty("length") Integer length,
                    @JsonProperty("set") Set<Integer> set) {
        if (set != null & op != ConditionOp.InSet && op != ConditionOp.NotInSet) {
            throw new IllegalArgumentException(
                            "Invalid condition op: can only use this constructor with InSet or NotInSet ops");
        }
        this.op = op;
        this.length = length;
        this.set = set;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema; //No op
    }

    @Override
    public String outputColumnName() {
        return inputSchema.getColumnNames().get(0);
    }

    @Override
    public String[] outputColumnNames() {
        return inputSchema.getColumnNames().toArray(new String[inputSchema.numColumns()]);
    }

    @Override
    public String[] columnNames() {
        return outputColumnNames();
    }

    @Override
    public String columnName() {
        return outputColumnName();
    }

    @Override
    public boolean condition(List<Writable> list) {
        throw new UnsupportedOperationException("Cannot apply SequenceLengthCondition on non-sequence data");
    }

    @Override
    public boolean condition(Object input) {
        throw new UnsupportedOperationException("Cannot apply SequenceLengthCondition on non-sequence data");
    }

    @Override
    public boolean conditionSequence(List<List<Writable>> sequence) {
        return op.apply(sequence.size(), (length == null ? 0 : length), set);
    }

    @Override
    public boolean conditionSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.inputSchema = schema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }
}
