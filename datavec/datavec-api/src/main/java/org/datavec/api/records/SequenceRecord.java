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

package org.datavec.api.records;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**
 * A SequenceRecord contains a set of values for a single sequence or time series (usually with multiple values per time step,
 * and multiple time steps).<br>
 * Each value in the Record is represented by {@link Writable} object; each time step is thus a {@code List<Writable>} and
 * the entire sequence is represented by a {@code List<List<Writable>>}, where the outer list is over time steps, and
 * the inner list is over values for a given time step.<br>
 * The SequenceRecord may (optionally) also have a {@link RecordMetaData} instance, that represents metadata (source
 * location, etc) for the record.<br>
 * For standard (non-sequential) data, see {@link Record}
 *
 * @author Alex Black
 */
public interface SequenceRecord extends Serializable {

    /**
     * Get the sequence record values
     *
     * @return Sequence record values
     */
    List<List<Writable>> getSequenceRecord();

    /**
     * Get the overall length of the sequence record (number of time/sequence steps, etc).
     * Equivalent to {@code getSequenceRecord().size()}
     *
     * @return Length of sequence record
     */
    int getSequenceLength();

    /**
     * Get a single time step. Equivalent to {@code getSequenceRecord().get(timeStep)}
     *
     * @param timeStep Time step to get. Must be {@code 0 <= timeStep < getSequenceLength()}
     * @return Values for a single time step
     */
    List<Writable> getTimeStep(int timeStep);

    /**
     * Set the sequence record values
     *
     * @param sequenceRecord Sequence record values to set
     */
    void setSequenceRecord(List<List<Writable>> sequenceRecord);

    /**
     * Get the RecordMetaData for this record
     *
     * @return Metadata for this record (or null, if none has been set)
     */
    RecordMetaData getMetaData();

    /**
     * Set the Record metadata
     */
    void setMetaData(RecordMetaData recordMetaData);

}
