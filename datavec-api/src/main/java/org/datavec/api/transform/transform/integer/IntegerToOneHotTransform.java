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

package org.datavec.api.transform.transform.integer;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Convert an integer column to a  set of one-hot columns.
 *
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"columnIdx"}, callSuper = false)
@JsonIgnoreProperties({"inputSchema", "columnIdx", "stateNames", "statesMap"})
public class IntegerToOneHotTransform extends BaseTransform {

    private String columnName;
    private int minValue;
    private int maxValue;
    private int columnIdx = -1;

    public IntegerToOneHotTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("minValue") int minValue, @JsonProperty("maxValue") int maxValue) {
        this.columnName = columnName;
        this.minValue = minValue;
        this.maxValue = maxValue;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        super.setInputSchema(inputSchema);

        columnIdx = inputSchema.getIndexOfColumn(columnName);
        ColumnMetaData meta = inputSchema.getMetaData(columnName);
        if (!(meta instanceof IntegerMetaData))
            throw new IllegalStateException("Cannot convert column \"" + columnName
                            + "\" from integer to one-hot: column is not integer (is: " + meta.getColumnType() + ")");
    }

    @Override
    public String toString() {
        return "CategoricalToOneHotTransform(columnName=\"" + columnName + "\")";

    }

    @Override
    public Schema transform(Schema schema) {
        List<String> origNames = schema.getColumnNames();
        List<ColumnMetaData> origMeta = schema.getColumnMetaData();

        int i = 0;
        Iterator<String> namesIter = origNames.iterator();
        Iterator<ColumnMetaData> typesIter = origMeta.iterator();

        List<ColumnMetaData> newMeta = new ArrayList<>(schema.numColumns());

        while (namesIter.hasNext()) {
            String s = namesIter.next();
            ColumnMetaData t = typesIter.next();

            if (i++ == columnIdx) {
                //Convert this to one-hot:
                for (int x = minValue; x <= maxValue; x++) {
                    String newName = s + "[" + x + "]";
                    newMeta.add(new IntegerMetaData(newName, 0, 1));
                }
            } else {
                newMeta.add(t);
            }
        }

        return schema.newSchema(newMeta);
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString());
        }
        int idx = getColumnIdx();

        int n = maxValue - minValue + 1;
        List<Writable> out = new ArrayList<>(writables.size() + n);

        int i = 0;
        for (Writable w : writables) {

            if (i++ == idx) {
                int currValue = w.toInt();
                if (currValue < minValue || currValue > maxValue) {
                    throw new IllegalStateException("Invalid value: integer value (" + currValue + ") is outside of "
                                    + "valid range: must be between " + minValue + " and " + maxValue + " inclusive");
                }

                for (int j = minValue; j <= maxValue; j++) {
                    if (j == currValue) {
                        out.add(new IntWritable(1));
                    } else {
                        out.add(new IntWritable(0));
                    }
                }
            } else {
                //No change to this column
                out.add(w);
            }
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
        int currValue = ((Number) input).intValue();
        if (currValue < minValue || currValue > maxValue) {
            throw new IllegalStateException("Invalid value: integer value (" + currValue + ") is outside of "
                            + "valid range: must be between " + minValue + " and " + maxValue + " inclusive");
        }

        List<Integer> oneHot = new ArrayList<>();
        for (int j = minValue; j <= maxValue; j++) {
            if (j == currValue) {
                oneHot.add(1);
            } else {
                oneHot.add(0);
            }
        }
        return oneHot;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        List<?> values = (List<?>) sequence;
        List<List<Integer>> ret = new ArrayList<>();
        for (Object obj : values) {
            ret.add((List<Integer>) map(obj));
        }
        return ret;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        throw new UnsupportedOperationException("Output column name will be more than 1");
    }

    /**
     * The output column names
     * This will often be the same as the input
     *
     * @return the output column names
     */
    @Override
    public String[] outputColumnNames() {
        List<String> l = transform(inputSchema).getColumnNames();
        return l.toArray(new String[l.size()]);
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
        return columnName;
    }
}
