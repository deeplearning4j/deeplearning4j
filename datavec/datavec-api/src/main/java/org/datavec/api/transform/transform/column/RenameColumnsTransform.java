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
import java.util.Collections;
import java.util.List;

/**
 * Rename one or more columns
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema"})
@Data
public class RenameColumnsTransform implements Transform, ColumnOp {

    private final List<String> oldNames;
    private final List<String> newNames;
    private Schema inputSchema;

    public RenameColumnsTransform(String oldName, String newName) {
        this(Collections.singletonList(oldName), Collections.singletonList(newName));
    }

    public RenameColumnsTransform(@JsonProperty("oldNames") List<String> oldNames,
                    @JsonProperty("newNames") List<String> newNames) {
        if (oldNames.size() != newNames.size())
            throw new IllegalArgumentException("Invalid input: old/new names lists differ in length");
        this.oldNames = oldNames;
        this.newNames = newNames;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        RenameColumnsTransform o2 = (RenameColumnsTransform) o;

        if (!oldNames.equals(o2.oldNames))
            return false;
        return newNames.equals(o2.newNames);

    }

    @Override
    public int hashCode() {
        int result = oldNames.hashCode();
        result = 31 * result + newNames.hashCode();
        return result;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        //Validate that all 'original' names exist:
        for( int i=0; i<oldNames.size(); i++ ){
            String s = oldNames.get(i);
            if(!inputSchema.hasColumn(s)){
                throw new IllegalStateException("Cannot rename from \"" + s + "\" to \"" + newNames.get(i)
                        + "\": original column name \"" + s + "\" does not exist. All columns for input schema: "
                        + inputSchema.getColumnNames());
            }
        }

        List<String> inputNames = inputSchema.getColumnNames();

        List<ColumnMetaData> outputMeta = new ArrayList<>();
        for (String s : inputNames) {
            int idx = oldNames.indexOf(s);
            if (idx >= 0) {
                //Switch the old and new names
                ColumnMetaData meta = inputSchema.getMetaData(s).clone();
                meta.setName(newNames.get(idx));
                outputMeta.add(meta);
            } else {
                outputMeta.add(inputSchema.getMetaData(s));
            }
        }

        return inputSchema.newSchema(outputMeta);
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        //No op
        return writables;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        //No op
        return sequence;
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
        throw new UnsupportedOperationException(
                        "Unable to map. Please treat this as a special operation. This should be handled by your implementation.");

    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException(
                        "Unable to map. Please treat this as a special operation. This should be handled by your implementation.");
    }

    @Override
    public String toString() {
        return "RenameColumnsTransform(oldNames=" + oldNames + ",newNames=" + newNames + ")";
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
        return newNames.toArray(new String[newNames.size()]);
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return oldNames.toArray(new String[oldNames.size()]);
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
