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
