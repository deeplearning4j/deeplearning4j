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

package org.datavec.api.records;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

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
