/*
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

package org.datavec.api.transform.transform;

import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import lombok.Data;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**Map the values in a single column to new values.
 * For example: string -> string, or empty -> x type transforms for a single column
 */
@Data
@JsonIgnoreProperties({"inputSchema","columnNumber"})
public abstract class BaseColumnTransform extends BaseTransform {

    protected final String columnName;
    protected int columnNumber = -1;
    private static final long serialVersionUID = 0L;

    public BaseColumnTransform(String columnName) {
        this.columnName = columnName;
    }

    @Override
    public void setInputSchema(Schema inputSchema){
        this.inputSchema = inputSchema;
        columnNumber = inputSchema.getIndexOfColumn(columnName);
    }

    @Override
    public Schema transform(Schema schema) {
        if(columnNumber == -1) throw new IllegalStateException("columnNumber == -1 -> setInputSchema not called?");
        List<ColumnMetaData> oldMeta = schema.getColumnMetaData();
        List<ColumnMetaData> newMeta = new ArrayList<>(oldMeta.size());

        Iterator<ColumnMetaData> typesIter = oldMeta.iterator();

        int i=0;
        while(typesIter.hasNext()){
            ColumnMetaData t = typesIter.next();
            if(i++ == columnNumber){
                newMeta.add(getNewColumnMetaData(t.getName(), t));
            } else {
                newMeta.add(t);
            }
        }

        return schema.newSchema(newMeta);
    }

    public abstract ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType);

    @Override
    public List<Writable> map(List<Writable> writables) {
        if(writables.size() != inputSchema.numColumns() ){
            throw new IllegalStateException("Cannot execute transform: input writables list length (" + writables.size() + ") does not " +
                    "match expected number of elements (schema: " + inputSchema.numColumns() + "). Transform = " + toString());
        }
        int n = writables.size();
        List<Writable> out = new ArrayList<>(n);

        int i=0;
        for(Writable w : writables){
            if(i++ == columnNumber){
                Writable newW = map(w);
                out.add(newW);
            } else {
                out.add(w);
            }
        }

        return out;
    }



    public abstract Writable map(Writable columnWritable);

    @Override
    public abstract String toString();

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        BaseColumnTransform o2 = (BaseColumnTransform) o;

        return columnName.equals(o2.columnName);

    }

    @Override
    public int hashCode() {
        return columnName.hashCode();
    }
}
