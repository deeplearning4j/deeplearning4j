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

package org.datavec.api.transform.transform.column;

import lombok.Data;
import org.datavec.api.transform.ColumnOp;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Duplicate one or more columns.
 * The duplicated columns
 * are placed immediately after the original columns
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"columnsToDuplicateSet", "columnIndexesToDuplicateSet", "inputSchema"})
@Data
public class DuplicateColumnsTransform implements Transform, ColumnOp {

    private final List<String> columnsToDuplicate;
    private final List<String> newColumnNames;
    private final Set<String> columnsToDuplicateSet;
    private final Set<Integer> columnIndexesToDuplicateSet;
    private Schema inputSchema;

    /**
     * @param columnsToDuplicate List of columns to duplicate
     * @param newColumnNames     List of names for the new (duplicate) columns
     */
    public DuplicateColumnsTransform(@JsonProperty("columnsToDuplicate") List<String> columnsToDuplicate,
                    @JsonProperty("newColumnNames") List<String> newColumnNames) {
        if (columnsToDuplicate == null || newColumnNames == null)
            throw new IllegalArgumentException("Columns/names cannot be null");
        if (columnsToDuplicate.size() != newColumnNames.size())
            throw new IllegalArgumentException(
                            "Invalid input: columns to duplicate and the new names must have equal lengths");
        this.columnsToDuplicate = columnsToDuplicate;
        this.newColumnNames = newColumnNames;
        this.columnsToDuplicateSet = new HashSet<>(columnsToDuplicate);
        this.columnIndexesToDuplicateSet = new HashSet<>();
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size() + newColumnNames.size());

        List<String> oldNames = inputSchema.getColumnNames();

        int dupCount = 0;
        for (int i = 0; i < oldMeta.size(); i++) {
            String current = oldNames.get(i);
            newMeta.add(oldMeta.get(i));

            if (columnsToDuplicateSet.contains(current)) {
                //Duplicate the current columnName, and place it after...
                String dupName = newColumnNames.get(dupCount);
                ColumnMetaData m = oldMeta.get(i).clone();
                m.setName(dupName);
                newMeta.add(m);
                dupCount++;
            }
        }

        return inputSchema.newSchema(newMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        columnIndexesToDuplicateSet.clear();

        List<String> schemaColumnNames = inputSchema.getColumnNames();
        for (String s : columnsToDuplicate) {
            int idx = schemaColumnNames.indexOf(s);
            if (idx == -1)
                throw new IllegalStateException("Invalid state: column to duplicate \"" + s + "\" does not appear "
                                + "in input schema");
            columnIndexesToDuplicateSet.add(idx);
        }

        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString());
        }
        List<Writable> out = new ArrayList<>(writables.size() + columnsToDuplicate.size());
        int i = 0;
        for (Writable w : writables) {
            out.add(w);
            if (columnIndexesToDuplicateSet.contains(i++))
                out.add(w); //TODO safter to copy here...
        }
        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>(sequence.size());
        for (List<Writable> l : sequence) {
            out.add(map(l));
        }
        return out;
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
        return input;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        return sequence;
    }

    @Override
    public String toString() {
        return "DuplicateColumnsTransform(toDuplicate=" + columnsToDuplicate + ",newNames=" + newColumnNames + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        DuplicateColumnsTransform o2 = (DuplicateColumnsTransform) o;

        if (!columnsToDuplicate.equals(o2.columnsToDuplicate))
            return false;
        return newColumnNames.equals(o2.newColumnNames);

    }

    @Override
    public int hashCode() {
        int result = columnsToDuplicate.hashCode();
        result = 31 * result + newColumnNames.hashCode();
        return result;
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
        return newColumnNames.toArray(new String[newColumnNames.size()]);
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return columnsToDuplicate.toArray(new String[columnsToDuplicate.size()]);
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
