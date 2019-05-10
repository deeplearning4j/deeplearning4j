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

package org.datavec.api.transform.sequence.comparator;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Sequence comparator: compare elements in a sequence using the values in a single column
 *
 * Can be applied on any numerical column (Integer, Long, Double or Time columns)
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"columnType", "schema", "columnIdx"})
@EqualsAndHashCode(callSuper = true, exclude = {"columnType"})
@Data
public class NumericalColumnComparator extends BaseColumnComparator {

    private ColumnType columnType;
    private boolean ascending;

    public NumericalColumnComparator(String columnName) {
        this(columnName, true);
    }

    public NumericalColumnComparator(@JsonProperty("columnName") String columnName,
                    @JsonProperty("ascending") boolean ascending) {
        super(columnName);
        this.ascending = ascending;
    }

    @Override
    public void setSchema(Schema sequenceSchema) {
        super.setSchema(sequenceSchema);
        this.columnType = sequenceSchema.getType(this.columnIdx);
        switch (columnType) {
            case Integer:
            case Long:
            case Double:
            case Time:
                //All ok. Time column uses LongWritables too...
                break;
            case Categorical:
            case Bytes:
            case String:
            default:
                throw new IllegalStateException(
                                "Cannot apply numerical column comparator on column of type " + columnType);
        }
    }

    @Override
    protected int compare(Writable w1, Writable w2) {
        int compare;
        switch (columnType) {
            case Integer:
                compare = Integer.compare(w1.toInt(), w2.toInt());
                break;
            case Time:
            case Long:
                compare = Long.compare(w1.toLong(), w2.toLong());
                break;
            case Double:
                compare = Double.compare(w1.toDouble(), w2.toDouble());
                break;
            default:
                //Should never happen...
                throw new RuntimeException("Cannot apply numerical column comparator on column of type " + columnType);
        }

        if (ascending)
            return compare;
        return -compare;
    }

    @Override
    public String toString() {
        return "NumericalColumnComparator(columnName=\"" + columnName + "\",ascending=" + ascending + ")";
    }
}
