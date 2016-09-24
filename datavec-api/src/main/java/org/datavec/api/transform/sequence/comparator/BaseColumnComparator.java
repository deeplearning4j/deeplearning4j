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

package org.datavec.api.transform.sequence.comparator;

import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.sequence.SequenceComparator;

import java.util.List;

/**
 * Compare/sort a sequence by the values of a specific column
 */
@EqualsAndHashCode(exclude = {"schema", "columnIdx"})
@JsonIgnoreProperties({"schema", "columnIdx"})
public abstract class BaseColumnComparator implements SequenceComparator {

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

    @Override
    public int compare(List<Writable> o1, List<Writable> o2) {
        return compare(get(o1, columnIdx), get(o2, columnIdx));
    }

    private static Writable get(List<Writable> c, int idx) {
        return c.get(idx);
    }

    protected abstract int compare(Writable w1, Writable w2);
}
