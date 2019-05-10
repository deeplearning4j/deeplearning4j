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

package org.datavec.api.transform.filter;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.*;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

/**
 * FilterInvalidValues: a filter operation that removes any examples (or sequences)
 * if the examples/sequences contains
 * invalid values in any of a specified set of columns.
 * Invalid values are determined with respect to the schema
 */
@EqualsAndHashCode(exclude = {"schema", "columnIdxs"})
@JsonIgnoreProperties({"schema", "columnIdxs"})
@Data
@ToString(exclude = {"schema", "columnIdxs"})
public class FilterInvalidValues implements Filter {

    private Schema schema;
    private final boolean filterAnyInvalid;
    private final String[] columnsToFilterIfInvalid;
    private int[] columnIdxs;

    /** Filter examples that have invalid values in ANY columns. */
    public FilterInvalidValues() {
        filterAnyInvalid = true;
        columnsToFilterIfInvalid = null;
    }

    /**
     * @param columnsToFilterIfInvalid Columns to check for invalid values
     */
    public FilterInvalidValues(String... columnsToFilterIfInvalid) {
        if (columnsToFilterIfInvalid == null || columnsToFilterIfInvalid.length == 0)
            throw new IllegalArgumentException("Cannot filter 0/null columns: columns to filter on must be specified");
        this.columnsToFilterIfInvalid = columnsToFilterIfInvalid;
        filterAnyInvalid = false;
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
        this.schema = schema;
        if (!filterAnyInvalid) {
            this.columnIdxs = new int[columnsToFilterIfInvalid.length];
            for (int i = 0; i < columnsToFilterIfInvalid.length; i++) {
                this.columnIdxs[i] = schema.getIndexOfColumn(columnsToFilterIfInvalid[i]);
            }
        }
    }

    @Override
    public Schema getInputSchema() {
        return schema;
    }

    /**
     * @param writables Example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeExample(Object writables) {
        List<?> row = (List<?>) writables;
        if (!filterAnyInvalid) {
            //Filter only on specific columns
            for (int i : columnIdxs) {
                if (filterColumn(row, i))
                    return true; //Remove if not valid

            }
        } else {
            //Filter on ALL columns
            int nCols = schema.numColumns();
            for (int i = 0; i < nCols; i++) {
                if (filterColumn(row, i))
                    return true;
            }
        }
        return false;
    }

    private boolean filterColumn(List<?> row, int i) {
        ColumnMetaData meta = schema.getMetaData(i);
        if (row.get(i) instanceof Float) {
            if (!meta.isValid(new FloatWritable((Float) row.get(i))))
                return true;
        } else if (row.get(i) instanceof Double) {
            if (!meta.isValid(new DoubleWritable((Double) row.get(i))))
                return true;
        } else if (row.get(i) instanceof String) {
            if (!meta.isValid(new Text(((String) row.get(i)).toString())))
                return true;
        } else if (row.get(i) instanceof Integer) {
            if (!meta.isValid(new IntWritable((Integer) row.get(i))))
                return true;

        } else if (row.get(i) instanceof Long) {
            if (!meta.isValid(new LongWritable((Long) row.get(i))))
                return true;
        } else if (row.get(i) instanceof Boolean) {
            if (!meta.isValid(new BooleanWritable((Boolean) row.get(i))))
                return true;
        }
        return false;
    }

    /**
     * @param sequence sequence example
     * @return true if example should be removed, false to keep
     */
    @Override
    public boolean removeSequence(Object sequence) {
        List<?> seq = (List<?>) sequence;
        //If _any_ of the values are invalid, remove the entire sequence
        for (Object c : seq) {
            if (removeExample(c))
                return true;
        }
        return false;
    }

    @Override
    public boolean removeExample(List<Writable> writables) {
        if (writables.size() != schema.numColumns())
            return true;

        if (!filterAnyInvalid) {
            //Filter only on specific columns
            for (int i : columnIdxs) {
                ColumnMetaData meta = schema.getMetaData(i);
                if (!meta.isValid(writables.get(i)))
                    return true; //Remove if not valid
            }
        } else {
            //Filter on ALL columns
            int nCols = schema.numColumns();
            for (int i = 0; i < nCols; i++) {
                ColumnMetaData meta = schema.getMetaData(i);
                if (!meta.isValid(writables.get(i)))
                    return true; //Remove if not valid
            }
        }
        return false;
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        //If _any_ of the values are invalid, remove the entire sequence
        for (List<Writable> c : sequence) {
            if (removeExample(c))
                return true;
        }
        return false;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
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
        return schema.getColumnNames().toArray(new String[schema.numColumns()]);
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
