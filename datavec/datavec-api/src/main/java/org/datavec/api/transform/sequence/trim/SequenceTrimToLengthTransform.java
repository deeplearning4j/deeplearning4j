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

package org.datavec.api.transform.sequence.trim;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.common.base.Preconditions;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Trim or pad the sequence to the specified length (number of sequence steps). It supports 2 modes:<br>
 * TRIM: Sequences longer than the specified maximum will be trimmed to exactly the maximum. Shorter sequences will not be modified.<br>
 * TRIM_OR_PAD: Sequences longer than the specified maximum will be trimmed to exactly the maximum. Shorter sequences will be
 * padded with as many copies of the "pad" array to make the sequence length equal the specified maximum.<br>
 * Note that the 'pad' list (i.e., values to pad when using TRIM_OR_PAD mode) must be equal in length to the number of columns (values per time step)
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"schema"})
@EqualsAndHashCode(exclude = {"schema"})
@Data
public class SequenceTrimToLengthTransform implements Transform {
    /**
     * Mode. See {@link SequenceTrimToLengthTransform}
     */
    public enum Mode {TRIM, TRIM_OR_PAD}

    private int maxLength;
    private Mode mode;
    private List<Writable> pad;

    private Schema schema;

    /**
     * @param maxLength maximum sequence length. Must be positive.
     * @param mode      Mode - trim or trim/pad
     * @param pad       Padding value. Only used with Mode.TRIM_OR_PAD. Must be equal in length to the number of columns (values per time step)
     */
    public SequenceTrimToLengthTransform(@JsonProperty("maxLength") int maxLength, @JsonProperty("mode") Mode mode, @JsonProperty("pad") List<Writable> pad) {
        Preconditions.checkState(maxLength > 0, "Maximum length must be > 0, got %s", maxLength);
        Preconditions.checkState(mode == Mode.TRIM || pad != null, "If mode == Mode.TRIM_OR_PAD ");
        this.maxLength = maxLength;
        this.mode = mode;
        this.pad = pad;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        throw new UnsupportedOperationException("SequenceTrimToLengthTransform cannot be applied to non-sequence values");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        if (mode == Mode.TRIM) {
            if (sequence.size() <= maxLength) {
                return sequence;
            }
            return new ArrayList<>(sequence.subList(0, maxLength));
        } else {
            //Trim or pad
            if (sequence.size() == maxLength) {
                return sequence;
            } else if (sequence.size() > maxLength) {
                return new ArrayList<>(sequence.subList(0, maxLength));
            } else {
                //Need to pad
                Preconditions.checkState(sequence.size() == 0 || sequence.get(0).size() == pad.size(), "Invalid padding values: %s padding " +
                        "values were provided, but data has %s values per time step (columns)", pad.size(), sequence.get(0).size());

                List<List<Writable>> out = new ArrayList<>(maxLength);
                out.addAll(sequence);
                while (out.size() < maxLength) {
                    out.add(pad);
                }
                return out;
            }
        }
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema;
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
        return null;
    }
}
