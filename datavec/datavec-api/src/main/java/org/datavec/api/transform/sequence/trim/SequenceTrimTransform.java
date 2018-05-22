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

package org.datavec.api.transform.sequence.trim;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * SequenceTrimTranform removes the first or last N values in a sequence. Note that the resulting sequence
 * may be of length 0, if the input sequence is less than or equal to N.
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"schema"})
@EqualsAndHashCode(exclude = {"schema"})
@Data
public class SequenceTrimTransform implements Transform {

    private int numStepsToTrim;
    private boolean trimFromStart;
    private Schema schema;

    /**
     *
     * @param numStepsToTrim Number of time steps to trim from the sequence
     * @param trimFromStart  If true: Trim values from the start of the sequence. If false: trim values from the end.
     */
    public SequenceTrimTransform(@JsonProperty("numStepsToTrim") int numStepsToTrim,
                    @JsonProperty("trimFromStart") boolean trimFromStart) {
        this.numStepsToTrim = numStepsToTrim;
        this.trimFromStart = trimFromStart;
    }

    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema; //No op
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.schema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return schema;
    }

    @Override
    public String outputColumnName() {
        return null;
    }

    @Override
    public String[] outputColumnNames() {
        return schema.getColumnNames().toArray(new String[schema.numColumns()]);
    }

    @Override
    public String[] columnNames() {
        return outputColumnNames();
    }

    @Override
    public String columnName() {
        return outputColumnName();
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        throw new UnsupportedOperationException("SequenceTrimTransform cannot be applied to non-sequence values");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        int start = 0;
        int end = sequence.size();
        if (trimFromStart) {
            start += numStepsToTrim;
        } else {
            end -= numStepsToTrim;
        }

        if (end < start) {
            return Collections.emptyList();
        }

        List<List<Writable>> out = new ArrayList<>(end - start);

        for (int i = start; i < end; i++) {
            out.add(sequence.get(i));
        }

        return out;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("SequenceTrimTransform cannot be applied to non-sequence values");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
