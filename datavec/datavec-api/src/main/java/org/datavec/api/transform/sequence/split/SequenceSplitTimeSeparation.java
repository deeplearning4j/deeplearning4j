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

package org.datavec.api.transform.sequence.split;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Split a sequence into multiple sequences, based on the separation of time steps in a time column.
 * For example, suppose we have a sequence with a gap of 1 day between two blocks of entries: we can use
 * SequenceSplitTimeSeparation to split this data into two separate sequences.
 *
 * More generally, split the sequence any time the separation between consecutive time steps exceeds a specified
 * value.
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"separationMilliseconds", "timeColumnIdx", "schema"})
@EqualsAndHashCode(exclude = {"separationMilliseconds", "timeColumnIdx", "schema"})
@Data
public class SequenceSplitTimeSeparation implements SequenceSplit {

    private final String timeColumn;
    private final long timeQuantity;
    private final TimeUnit timeUnit;
    private final long separationMilliseconds;
    private int timeColumnIdx = -1;
    private Schema schema;

    /**
     * @param timeColumn      Time column to consider when splitting
     * @param timeQuantity    Value/amount (of the specified TimeUnit)
     * @param timeUnit        The unit of time
     */
    public SequenceSplitTimeSeparation(@JsonProperty("timeColumn") String timeColumn,
                    @JsonProperty("timeQuantity") long timeQuantity, @JsonProperty("timeUnit") TimeUnit timeUnit) {
        this.timeColumn = timeColumn;
        this.timeQuantity = timeQuantity;
        this.timeUnit = timeUnit;

        this.separationMilliseconds = TimeUnit.MILLISECONDS.convert(timeQuantity, timeUnit);
    }

    @Override
    public List<List<List<Writable>>> split(List<List<Writable>> sequence) {

        List<List<List<Writable>>> out = new ArrayList<>();

        long lastTimeStepTime = Long.MIN_VALUE;
        List<List<Writable>> currentSplit = null;

        for (List<Writable> timeStep : sequence) {
            long currStepTime = timeStep.get(timeColumnIdx).toLong();
            if (lastTimeStepTime == Long.MIN_VALUE || (currStepTime - lastTimeStepTime) > separationMilliseconds) {
                //New split
                if (currentSplit != null)
                    out.add(currentSplit);
                currentSplit = new ArrayList<>();
            }
            currentSplit.add(timeStep);
            lastTimeStepTime = currStepTime;
        }

        //Add the final split to the output...
        out.add(currentSplit);

        return out;
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        if (!inputSchema.hasColumn(timeColumn))
            throw new IllegalStateException(
                            "Invalid state: schema does not have column " + "with name \"" + timeColumn + "\"");
        if (inputSchema.getMetaData(timeColumn).getColumnType() != ColumnType.Time) {
            throw new IllegalStateException("Invalid input schema: schema column \"" + timeColumn
                            + "\" is not a time column." + " (Is type: "
                            + inputSchema.getMetaData(timeColumn).getColumnType() + ")");
        }

        this.timeColumnIdx = inputSchema.getIndexOfColumn(timeColumn);
        this.schema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return schema;
    }

    @Override
    public String toString() {
        return "SequenceSplitTimeSeparation(timeColumn=\"" + timeColumn + "\",timeQuantity=" + timeQuantity
                        + ",timeUnit=" + timeUnit + ")";
    }
}
