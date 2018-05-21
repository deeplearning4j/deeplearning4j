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
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.BaseTransform;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import org.nd4j.util.StringUtils;

import java.util.*;

/**
 * Remove the specified columns from the data.
 * To specify only the columns to keep,
 * use {@link RemoveAllColumnsExceptForTransform}
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "columnsToRemoveIdx", "indicesToRemove"})
@Data
public class RemoveColumnsTransform extends BaseTransform implements ColumnOp {

    private int[] columnsToRemoveIdx;
    private String[] columnsToRemove;
    private Set<Integer> indicesToRemove;
    private String[] leftOverColumns;

    public RemoveColumnsTransform(@JsonProperty("columnsToRemove") String... columnsToRemove) {
        this.columnsToRemove = columnsToRemove;

    }

    @Override
    public void setInputSchema(Schema schema) {
        super.setInputSchema(schema);
        //Validate that all 'columns to be removed exist
        for(String s : columnsToRemove){
            if(!inputSchema.hasColumn(s)){
                throw new IllegalStateException("Cannot remove column \"" + s + "\": column does not exist. All " +
                        "columns for input schema: " + inputSchema.getColumnNames());
            }
        }


        leftOverColumns = new String[schema.numColumns() - columnsToRemove.length];

        indicesToRemove = new HashSet<>();

        int i = 0;
        columnsToRemoveIdx = new int[columnsToRemove.length];
        for (String s : columnsToRemove) {
            int idx = schema.getIndexOfColumn(s);
            if (idx < 0)
                throw new RuntimeException("Column \"" + s + "\" not found");
            columnsToRemoveIdx[i++] = idx;
            indicesToRemove.add(idx);
        }


        int leftOverColumnsIdx = 0;
        List<String> columnTest = Arrays.asList(columnsToRemove);
        List<String> origColumnNames = schema.getColumnNames();
        for (int remove = 0; remove < schema.numColumns(); remove++) {
            if (!columnTest.contains(origColumnNames.get(remove)))
                leftOverColumns[leftOverColumnsIdx++] = origColumnNames.get(remove);
        }
    }

    @Override
    public Schema transform(Schema schema) {
        int nToRemove = columnsToRemove.length;
        int newNumColumns = schema.numColumns() - nToRemove;
        if (newNumColumns <= 0)
            throw new IllegalStateException("Number of columns after executing operation is " + newNumColumns
                            + " (is <= 0). " + "origColumns = " + schema.getColumnNames() + ", toRemove = "
                            + Arrays.toString(columnsToRemove));

        List<String> origNames = schema.getColumnNames();
        List<ColumnMetaData> origMeta = schema.getColumnMetaData();

        Set<String> set = new HashSet<>();
        Collections.addAll(set, columnsToRemove);


        List<ColumnMetaData> newMeta = new ArrayList<>(newNumColumns);

        Iterator<String> namesIter = origNames.iterator();
        Iterator<ColumnMetaData> metaIter = origMeta.iterator();

        while (namesIter.hasNext()) {
            String n = namesIter.next();
            ColumnMetaData t = metaIter.next();
            if (!set.contains(n)) {
                newMeta.add(t);
            }
        }

        return schema.newSchema(newMeta);
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        if (writables.size() != inputSchema.numColumns()) {
            List<String> list = new ArrayList<>();
            for (Writable w : writables)
                list.add(w.toString());
            String toString = StringUtils.join(",", list);
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size()
                            + ") does not " + "match expected number of elements (schema: " + inputSchema.numColumns()
                            + "). Transform = " + toString() + " and record " + toString);
        }

        List<Writable> outList = new ArrayList<>(writables.size() - columnsToRemove.length);

        int i = 0;
        for (Writable w : writables) {
            if (indicesToRemove.contains(i++))
                continue;
            outList.add(w);
        }
        return outList;
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
        return "RemoveColumnsTransform(" + Arrays.toString(columnsToRemove) + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        RemoveColumnsTransform o2 = (RemoveColumnsTransform) o;

        return Arrays.equals(columnsToRemove, o2.columnsToRemove);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(columnsToRemove);
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
        return leftOverColumns;
    }

    /**
     * Returns column names
     * this op is meant to run on
     *
     * @return
     */
    @Override
    public String[] columnNames() {
        return inputSchema.getColumnNames().toArray(new String[inputSchema.numColumns()]);
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
