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

package org.datavec.api.transform.transform.sequence;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;

/**
 * Sequence offset transform takes a sequence, and shifts The values in one or more columns by a specified number of
 * times steps. It has 2 modes of operation (OperationType enum), with respect to the columns it operates on:<br>
 * InPlace: operations may be performed in-place, modifying the values in the specified columns<br>
 * NewColumn: operations may produce new columns, with the original (source) columns remaining unmodified<br>
 * <p>
 * Additionally, there are 2 modes for handling values outside the original sequence (EdgeHandling enum):
 * TrimSequence: the entire sequence is trimmed (start or end) by a specified number of steps<br>
 * SpecifiedValue: for any values outside of the original sequence, they are given a specified value<br>
 * <p>
 * Note 1: When specifying offsets, they are done as follows:
 * Positive offsets: move the values in the specified columns to a later time. Earlier time steps are either be trimmed
 * or Given specified values; the last values in these columns will be truncated/removed.
 * <p>
 * Note 2: Care must be taken when using TrimSequence: for example, if we chain multiple sequence offset transforms on the
 * one dataset, we may end up trimming much more than we want. In this case, it may be better to use SpecifiedValue,
 * (with, for example, NullWritable) and then do a single trim operation (via {@link org.datavec.api.transform.sequence.trim.SequenceTrimTransform})
 * at the end.
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema", "columnsToOffsetSet"})
@JsonInclude(JsonInclude.Include.NON_NULL)
@Data
@EqualsAndHashCode(exclude = {"columnsToOffsetSet", "inputSchema"})
public class SequenceOffsetTransform implements Transform {

    public enum OperationType {
        InPlace, NewColumn
    }

    public enum EdgeHandling {
        TrimSequence, SpecifiedValue
    }

    private List<String> columnsToOffset;
    private int offsetAmount;
    private OperationType operationType;
    private EdgeHandling edgeHandling;
    private Writable edgeCaseValue;

    private Set<String> columnsToOffsetSet;
    @Getter
    private Schema inputSchema;

    public SequenceOffsetTransform(@JsonProperty("columnsToOffset") List<String> columnsToOffset,
                    @JsonProperty("offsetAmount") int offsetAmount,
                    @JsonProperty("operationType") OperationType operationType,
                    @JsonProperty("edgeHandling") EdgeHandling edgeHandling,
                    @JsonProperty("edgeCaseValue") Writable edgeCaseValue) {
        if (edgeCaseValue != null && edgeHandling != EdgeHandling.SpecifiedValue) {
            throw new UnsupportedOperationException(
                            "edgeCaseValue was non-null, but EdgeHandling was not set to SpecifiedValue. "
                                            + "edgeCaseValue can only be used with SpecifiedValue mode");
        }

        this.columnsToOffset = columnsToOffset;
        this.offsetAmount = offsetAmount;
        this.operationType = operationType;
        this.edgeHandling = edgeHandling;
        this.edgeCaseValue = edgeCaseValue;

        this.columnsToOffsetSet = new HashSet<>(columnsToOffset);
    }

    @Override
    public Schema transform(Schema inputSchema) {
        for (String s : columnsToOffset) {
            if (!inputSchema.hasColumn(s)) {
                throw new IllegalStateException("Column \"" + s + "\" is not found in input schema");
            }
        }

        List<ColumnMetaData> newMeta = new ArrayList<>();
        for (ColumnMetaData m : inputSchema.getColumnMetaData()) {
            if (columnsToOffsetSet.contains(m.getName())) {
                if (operationType == OperationType.InPlace) {
                    //Only change is to the name
                    ColumnMetaData mNew = m.clone();
                    mNew.setName(getNewColumnName(m));
                } else {
                    //Original is unmodified, new column is added
                    newMeta.add(m);
                    ColumnMetaData mNew = m.clone();
                    mNew.setName(getNewColumnName(m));
                    newMeta.add(mNew);
                }
            } else {
                //No change to this column
                newMeta.add(m);
            }
        }

        return inputSchema.newSchema(newMeta);
    }

    private String getNewColumnName(ColumnMetaData m) {
        return "sequenceOffset(" + offsetAmount + "," + m.getName() + ")";
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public String outputColumnName() {
        return outputColumnNames()[0];
    }

    @Override
    public String[] outputColumnNames() {
        return inputSchema.getColumnNames().toArray(new String[inputSchema.numColumns()]);
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
        throw new UnsupportedOperationException("SequenceOffsetTransform cannot be applied to non-sequence data");
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        //Edge case
        if (offsetAmount >= sequence.size() && edgeHandling == EdgeHandling.TrimSequence) {
            //No output
            return Collections.emptyList();
        }

        List<String> colNames = inputSchema.getColumnNames();
        int nIn = inputSchema.numColumns();
        int nOut = nIn + (operationType == OperationType.InPlace ? 0 : columnsToOffset.size());

        //Depending on settings, the original sequence might be smaller than the input
        int firstOutputStepInclusive;
        int lastOutputStepInclusive;
        if (edgeHandling == EdgeHandling.TrimSequence) {
            if (offsetAmount >= 0) {
                //Values in the specified columns are shifted later -> trim the start of the sequence
                firstOutputStepInclusive = offsetAmount;
                lastOutputStepInclusive = sequence.size() - 1;
            } else {
                //Values in the specified columns are shifted earlier -> trim the end of the sequence
                firstOutputStepInclusive = 0;
                lastOutputStepInclusive = sequence.size() - 1 + offsetAmount;
            }
        } else {
            //Specified value -> same output size
            firstOutputStepInclusive = 0;
            lastOutputStepInclusive = sequence.size() - 1;
        }

        List<List<Writable>> out = new ArrayList<>();
        for (int step = firstOutputStepInclusive; step <= lastOutputStepInclusive; step++) {
            List<Writable> thisStepIn = sequence.get(step); //Input for the *non-shifted* values
            List<Writable> thisStepOut = new ArrayList<>(nOut);



            for (int j = 0; j < nIn; j++) {
                if (columnsToOffsetSet.contains(colNames.get(j))) {

                    if (edgeHandling == EdgeHandling.SpecifiedValue && step - offsetAmount < 0
                                    || step - offsetAmount >= sequence.size()) {
                        if (operationType == OperationType.NewColumn) {
                            //Keep the original value
                            thisStepOut.add(thisStepIn.get(j));
                        }
                        thisStepOut.add(edgeCaseValue);
                    } else {
                        //Trim case, or specified but within range
                        Writable shifted = sequence.get(step - offsetAmount).get(j);
                        if (operationType == OperationType.InPlace) {
                            //Shift by the specified amount and output
                            thisStepOut.add(shifted);
                        } else {
                            //Add the old value and the new (offset) value
                            thisStepOut.add(thisStepIn.get(j));
                            thisStepOut.add(shifted);
                        }
                    }
                } else {
                    //Value is unmodified in this column
                    thisStepOut.add(thisStepIn.get(j));
                }
            }

            out.add(thisStepOut);
        }

        return out;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("SequenceOffsetTransform cannot be applied to non-sequence data");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented/supported");
    }
}
