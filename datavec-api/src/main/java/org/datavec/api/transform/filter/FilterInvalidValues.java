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

package org.datavec.api.transform.filter;

import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * FilterInvalidValues: a filter operation that removes any examples (or sequences) if the examples/sequences contains
 * invalid values in any of a specified set of columns.
 * Invalid values are determined with respect to the schema
 */
@EqualsAndHashCode(exclude = {"schema","columnIdxs"})
@JsonIgnoreProperties({"schema","columnIdxs"})
public class FilterInvalidValues implements Filter {

    private Schema schema;
    private final boolean filterAnyInvalid;
    private final String[] columnsToFilterIfInvalid;
    private int[] columnIdxs;

    /** Filter examples that have invalid values in ANY columns. */
    public FilterInvalidValues(){
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

    @Override
    public void setInputSchema(Schema schema) {
        this.schema = schema;
        if(!filterAnyInvalid) {
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

    @Override
    public boolean removeExample(List<Writable> writables) {
        if(writables.size() != schema.numColumns()) return true;

        if(!filterAnyInvalid) {
            //Filter only on specific columns
            for (int i : columnIdxs) {
                ColumnMetaData meta = schema.getMetaData(i);
                if (!meta.isValid(writables.get(i))) return true; //Remove if not valid
            }
        } else {
            //Filter or ALL columns
            int nCols = schema.numColumns();
            for( int i=0; i<nCols; i++ ){
                ColumnMetaData meta = schema.getMetaData(i);
                if (!meta.isValid(writables.get(i))) return true; //Remove if not valid
            }
        }
        return false;
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        //If _any_ of the values are invalid, remove the entire sequence
        for (List<Writable> c : sequence) {
            if (removeExample(c)) return true;
        }
        return false;
    }
}
