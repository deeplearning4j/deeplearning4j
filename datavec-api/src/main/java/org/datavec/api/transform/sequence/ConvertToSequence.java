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

package org.datavec.api.transform.sequence;


import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;

import java.util.Arrays;
import java.util.Collection;

/**
 * Convert a set of values to a sequence
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonIgnoreProperties({"inputSchema"})
public class ConvertToSequence {

    private final String[] keyColumns;
    private final SequenceComparator comparator; //For sorting values within collected (unsorted) sequence
    private Schema inputSchema;

    /**
     *
     * @param keyColumn The value to use as the key for inferring which examples belong to what sequence
     * @param comparator The comparator to use when deciding the order of each possible time step in the sequence
     */
    public ConvertToSequence(String keyColumn, SequenceComparator comparator){
        this(new String[]{keyColumn}, comparator);
    }

    /**
     *
     * @param keyColumns The value or values to use as the key (multiple values: compound key)  for inferring which
     *                   examples belong to what sequence
     * @param comparator The comparator to use when deciding the order of each possible time step in the sequence
     */
    public ConvertToSequence(Collection<String> keyColumns, SequenceComparator comparator) {
        this(keyColumns.toArray(new String[keyColumns.size()]), comparator);
    }

    /**
     *
     * @param keyColumns The value or values to use as the key (multiple values: compound key)  for inferring which
     *                   examples belong to what sequence
     * @param comparator The comparator to use when deciding the order of each possible time step in the sequence
     */
    public ConvertToSequence(@JsonProperty("keyColumn") String[] keyColumns,
                    @JsonProperty("comparator") SequenceComparator comparator) {
        this.keyColumns = keyColumns;
        this.comparator = comparator;
    }

    public SequenceSchema transform(Schema schema) {
        return new SequenceSchema(schema.getColumnMetaData());
    }

    public void setInputSchema(Schema schema) {
        this.inputSchema = schema;
        comparator.setSchema(transform(schema));
    }

    @Override
    public String toString() {
        if(keyColumns.length == 1){
            return "ConvertToSequence(keyColumn=\"" + keyColumns[0] + "\",comparator=" + comparator + ")";
        } else {
            return "ConvertToSequence(keyColumns=\"" + Arrays.toString(keyColumns) + "\",comparator=" + comparator + ")";
        }
    }

}
