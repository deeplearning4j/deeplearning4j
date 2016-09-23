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

package org.datavec.api.transform.sequence;


import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;

/**
 * Convert a set of values to a sequence
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonIgnoreProperties({"inputSchema"})
public class ConvertToSequence {

    private final String keyColumn;
    private final SequenceComparator comparator;    //For sorting values within collected (unsorted) sequence
    private Schema inputSchema;

    public ConvertToSequence(@JsonProperty("keyColumn") String keyColumn, @JsonProperty("comparator") SequenceComparator comparator) {
        this.keyColumn = keyColumn;
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
        return "ConvertToSequence(keyColumn=\"" + keyColumn + "\",comparator=" + comparator + ")";
    }

}
