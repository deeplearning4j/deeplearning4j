/*-
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.transform.transform.categorical;

import lombok.Data;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.*;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Pivot transform operates on two columns:
 * - a categorical column that operates as a key, and
 * - Another column that contains a value
 * Essentially, Pivot transform takes keyvalue pairs and breaks them out into separate columns.
 *
 * For example, with schema [col0, key, value, col3]
 * and values with key in {a,b,c}
 * Output schema is [col0, key[a], key[b], key[c], col3]
 * and input (col0Val, b, x, col3Val) gets mapped to (col0Val, 0, x, 0, col3Val).
 *
 * When expanding columns, a default value is used - for example 0 for numerical columns.
 *
 * @author Alex Black
 */
@Data
public class PivotTransform extends BaseTransform {

    private final String keyColumn;
    private final String valueColumn;
    private Writable defaultValue;

    /**
     *
     * @param keyColumnName   Key column to expand
     * @param valueColumnName Name of the column that contains the value
     */
    public PivotTransform(String keyColumnName, String valueColumnName) {
        this(keyColumnName, valueColumnName, null);
    }

    /**
     *
     * @param keyColumnName   Key column to expand
     * @param valueColumnName Name of the column that contains the value
     * @param defaultValue    The default value to use in expanded columns.
     */
    public PivotTransform(String keyColumnName, String valueColumnName, Writable defaultValue) {
        this.keyColumn = keyColumnName;
        this.valueColumn = valueColumnName;
        this.defaultValue = defaultValue;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        if (!inputSchema.hasColumn(keyColumn) || !inputSchema.hasColumn(valueColumn)) {
            throw new UnsupportedOperationException("Key or value column not found: " + keyColumn + ", " + valueColumn
                            + " in " + inputSchema.getColumnNames());
        }

        List<String> origNames = inputSchema.getColumnNames();
        List<ColumnMetaData> origMeta = inputSchema.getColumnMetaData();

        int i = 0;
        Iterator<String> namesIter = origNames.iterator();
        Iterator<ColumnMetaData> typesIter = origMeta.iterator();

        List<ColumnMetaData> newMeta = new ArrayList<>(inputSchema.numColumns());

        int idxKey = inputSchema.getIndexOfColumn(keyColumn);
        int idxValue = inputSchema.getIndexOfColumn(valueColumn);

        ColumnMetaData valueMeta = inputSchema.getMetaData(idxValue);

        while (namesIter.hasNext()) {
            String s = namesIter.next();
            ColumnMetaData t = typesIter.next();

            if (i == idxKey) {
                //Convert this to a set of separate columns
                List<String> stateNames = ((CategoricalMetaData) inputSchema.getMetaData(idxKey)).getStateNames();
                for (String stateName : stateNames) {
                    String newName = s + "[" + stateName + "]";

                    ColumnMetaData newValueMeta = valueMeta.clone();
                    newValueMeta.setName(newName);

                    newMeta.add(newValueMeta);
                }
            } else if (i == idxValue) {
                i++;
                continue; //Skip column
            } else {
                newMeta.add(t);
            }
            i++;
        }

        //Infer the default value if necessary
        if (defaultValue == null) {
            switch (valueMeta.getColumnType()) {
                case String:
                    defaultValue = new Text("");
                    break;
                case Integer:
                    defaultValue = new IntWritable(0);
                    break;
                case Long:
                    defaultValue = new LongWritable(0);
                    break;
                case Double:
                    defaultValue = new DoubleWritable(0.0);
                    break;
                case Float:
                    defaultValue = new FloatWritable(0.0f);
                    break;
                case Categorical:
                    defaultValue = new NullWritable();
                    break;
                case Time:
                    defaultValue = new LongWritable(0);
                    break;
                case Bytes:
                    throw new UnsupportedOperationException("Cannot infer default value for bytes");
                case Boolean:
                    defaultValue = new Text("false");
                    break;
                default:
                    throw new UnsupportedOperationException(
                                    "Cannot infer default value for " + valueMeta.getColumnType());
            }
        }

        return inputSchema.newSchema(newMeta);
    }


    @Override
    public String outputColumnName() {
        throw new UnsupportedOperationException("Output column name will be more than 1");
    }

    @Override
    public String[] outputColumnNames() {
        List<String> l = ((CategoricalMetaData) inputSchema.getMetaData(keyColumn)).getStateNames();
        return l.toArray(new String[l.size()]);
    }

    @Override
    public String[] columnNames() {
        return new String[] {keyColumn, valueColumn};
    }

    @Override
    public String columnName() {
        throw new UnsupportedOperationException("Multiple input columns");
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString());
        }

        int idxKey = inputSchema.getIndexOfColumn(keyColumn);
        int idxValue = inputSchema.getIndexOfColumn(valueColumn);
        List<String> stateNames = ((CategoricalMetaData) inputSchema.getMetaData(idxKey)).getStateNames();

        int i = 0;
        List<Writable> out = new ArrayList<>();
        for (Writable w : writables) {

            if (i == idxKey) {
                //Do conversion
                String str = w.toString();
                int stateIdx = stateNames.indexOf(str);

                if (stateIdx < 0)
                    throw new RuntimeException("Unknown state (index not found): " + str);
                for (int j = 0; j < stateNames.size(); j++) {
                    if (j == stateIdx) {
                        out.add(writables.get(idxValue));
                    } else {
                        out.add(defaultValue);
                    }
                }
            } else if (i == idxValue) {
                i++;
                continue;
            } else {
                //No change to this column
                out.add(w);
            }
            i++;
        }
        return out;
    }

    @Override
    public Object map(Object input) {
        List<Writable> l = (List<Writable>) input;
        Writable k = l.get(0);
        Writable v = l.get(1);

        int idxKey = inputSchema.getIndexOfColumn(keyColumn);
        List<String> stateNames = ((CategoricalMetaData) inputSchema.getMetaData(idxKey)).getStateNames();
        int n = stateNames.size();

        int position = stateNames.indexOf(k.toString());

        List<Writable> out = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            if (j == position) {
                out.add(v);
            } else {
                out.add(defaultValue);
            }
        }
        return out;
    }

    @Override
    public Object mapSequence(Object sequence) {
        List<?> values = (List<?>) sequence;
        List<List<Integer>> ret = new ArrayList<>();
        for (Object obj : values) {
            ret.add((List<Integer>) map(obj));
        }
        return ret;
    }
}
