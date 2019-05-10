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

package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * Convert a delimited String to a list of binary categorical columns.
 * Suppose the possible String values were {"a","b","c","d"} and the String column value to be converted contained
 * the String "a,c", then the 4 output columns would have values ["true","false","true","false"]
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "map", "columnIdx"})
@EqualsAndHashCode(callSuper = false, exclude = {"columnIdx"})
@Data
public class StringListToCategoricalSetTransform extends BaseTransform {

    private final String columnName;
    private final List<String> newColumnNames;
    private final List<String> categoryTokens;
    private final String delimiter;

    private final Map<String, Integer> map;

    private int columnIdx = -1;

    /**
     * @param columnName     The name of the column to convert
     * @param newColumnNames The names of the new columns to create
     * @param categoryTokens The possible tokens that may be present. Note this list must have the same length and order
     *                       as the newColumnNames list
     * @param delimiter      The delimiter for the Strings to convert
     */
    public StringListToCategoricalSetTransform(@JsonProperty("columnName") String columnName,
                    @JsonProperty("newColumnNames") List<String> newColumnNames,
                    @JsonProperty("categoryTokens") List<String> categoryTokens,
                    @JsonProperty("delimiter") String delimiter) {
        if (newColumnNames.size() != categoryTokens.size())
            throw new IllegalArgumentException("Names/tokens sizes cannot differ");
        this.columnName = columnName;
        this.newColumnNames = newColumnNames;
        this.categoryTokens = categoryTokens;
        this.delimiter = delimiter;

        map = new HashMap<>();
        for (int i = 0; i < categoryTokens.size(); i++) {
            map.put(categoryTokens.get(i), i);
        }
    }

    @Override
    public Schema transform(Schema inputSchema) {

        int colIdx = inputSchema.getIndexOfColumn(columnName);

        List<ColumnMetaData> oldMeta = inputSchema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size() + newColumnNames.size() - 1);
        List<String> oldNames = inputSchema.getColumnNames();

        Iterator<ColumnMetaData> typesIter = oldMeta.iterator();
        Iterator<String> namesIter = oldNames.iterator();

        int i = 0;
        while (typesIter.hasNext()) {
            ColumnMetaData t = typesIter.next();
            String name = namesIter.next();
            if (i++ == colIdx) {
                //Replace String column with a set of binary/categorical columns
                if (t.getColumnType() != ColumnType.String)
                    throw new IllegalStateException("Cannot convert non-string type");

                for (int j = 0; j < newColumnNames.size(); j++) {
                    ColumnMetaData meta = new CategoricalMetaData(newColumnNames.get(j), "true", "false");
                    newMeta.add(meta);
                }
            } else {
                newMeta.add(t);
            }
        }

        return inputSchema.newSchema(newMeta);

    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
        this.columnIdx = inputSchema.getIndexOfColumn(columnName);
    }

    @Override
    public String toString() {
        return "StringListToCategoricalSetTransform(columnName=" + columnName + ",newColumnNames=" + newColumnNames
                        + ",categoryTokens=" + categoryTokens + ",delimiter=\"" + delimiter + "\")";
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString());
        }
        int n = writables.size();
        List<Writable> out = new ArrayList<>(n);

        int i = 0;
        for (Writable w : writables) {
            if (i++ == columnIdx) {
                String str = w.toString();
                boolean[] present = new boolean[categoryTokens.size()];
                if (str != null && !str.isEmpty()) {
                    String[] split = str.split(delimiter);
                    for (String s : split) {
                        Integer idx = map.get(s);
                        if (idx == null)
                            throw new IllegalStateException("Encountered unknown String: \"" + s + "\"");
                        present[idx] = true;
                    }
                }
                for (int j = 0; j < present.length; j++) {
                    out.add(new Text(present[j] ? "true" : "false"));
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
        return null;
    }

    /**
     * Transform a sequence
     *
     * @param sequence
     */
    @Override
    public Object mapSequence(Object sequence) {
        return null;
    }

    /**
     * The output column name
     * after the operation has been applied
     *
     * @return the output column name
     */
    @Override
    public String outputColumnName() {
        throw new UnsupportedOperationException("New column names is always more than 1 in length");
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
        return new String[] {columnName()};
    }

    /**
     * Returns a singular column name
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String columnName() {
        return columnName();
    }
}
