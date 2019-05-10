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

package org.datavec.api.transform.sequence.comparator;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

/**
 * Compare/sort a sequence by the values of a specific column
 */
@EqualsAndHashCode(exclude = {"schema", "columnIdx"})
@JsonIgnoreProperties({"schema", "columnIdx"})
@Data
public abstract class BaseColumnComparator implements SequenceComparator, ColumnOp {

    protected Schema schema;

    protected final String columnName;
    protected int columnIdx = -1;

    protected BaseColumnComparator(String columnName) {
        this.columnName = columnName;
    }

    @Override
    public void setSchema(Schema sequenceSchema) {
        this.schema = sequenceSchema;
        this.columnIdx = sequenceSchema.getIndexOfColumn(columnName);
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

    /**
     * Set the input schema.
     *
     * @param inputSchema
     */
    @Override
    public void setInputSchema(Schema inputSchema) {
        this.schema = inputSchema;
    }

    /**
     * Getter for input schema
     *
     * @return
     */
    @Override
    public Schema getInputSchema() {
        return schema;
    }

    @Override
    public int compare(List<Writable> o1, List<Writable> o2) {
        return compare(get(o1, columnIdx), get(o2, columnIdx));
    }

    private static Writable get(List<Writable> c, int idx) {
        return c.get(idx);
    }

    protected abstract int compare(Writable w1, Writable w2);

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
}
