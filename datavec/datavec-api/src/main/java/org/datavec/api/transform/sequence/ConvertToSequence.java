/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.api.transform.sequence;


import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Collection;

@Data
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonIgnoreProperties({"inputSchema"})
public class ConvertToSequence {

    private boolean singleStepSequencesMode;
    private final String[] keyColumns;
    private final SequenceComparator comparator; //For sorting values within collected (unsorted) sequence
    private Schema inputSchema;

    /**
     *
     * @param keyColumn The value to use as the key for inferring which examples belong to what sequence
     * @param comparator The comparator to use when deciding the order of each possible time step in the sequence
     */
    public ConvertToSequence(String keyColumn, SequenceComparator comparator){
        this(false, new String[]{keyColumn}, comparator);
    }

    /**
     *
     * @param keyColumns The value or values to use as the key (multiple values: compound key)  for inferring which
     *                   examples belong to what sequence
     * @param comparator The comparator to use when deciding the order of each possible time step in the sequence
     */
    public ConvertToSequence(Collection<String> keyColumns, SequenceComparator comparator) {
        this(false, keyColumns.toArray(new String[keyColumns.size()]), comparator);
    }

    /**
     *
     * @param keyColumns The value or values to use as the key (multiple values: compound key)  for inferring which
     *                   examples belong to what sequence
     * @param comparator The comparator to use when deciding the order of each possible time step in the sequence
     */
    public ConvertToSequence(@JsonProperty("singleStepSequencesMode") boolean singleStepSequencesMode,
                             @JsonProperty("keyColumn") String[] keyColumns,
                             @JsonProperty("comparator") SequenceComparator comparator) {
        this.singleStepSequencesMode = singleStepSequencesMode;
        this.keyColumns = keyColumns;
        this.comparator = comparator;
    }

    public SequenceSchema transform(Schema schema) {
        return new SequenceSchema(schema.getColumnMetaData());
    }

    public void setInputSchema(Schema schema) {
        this.inputSchema = schema;
        if(!singleStepSequencesMode){
            comparator.setSchema(transform(schema));
        }
    }

    @Override
    public String toString() {
        if(singleStepSequencesMode) {
            return "ConvertToSequence()";
        } else if(keyColumns.length == 1){
            return "ConvertToSequence(keyColumn=\"" + keyColumns[0] + "\",comparator=" + comparator + ")";
        } else {
            return "ConvertToSequence(keyColumns=\"" + Arrays.toString(keyColumns) + "\",comparator=" + comparator + ")";
        }
    }

}
