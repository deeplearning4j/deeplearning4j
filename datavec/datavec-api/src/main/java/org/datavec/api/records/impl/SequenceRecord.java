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

package org.datavec.api.records.impl;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * A standard implementation of the {@link org.datavec.api.records.SequenceRecord} interface.
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class SequenceRecord implements org.datavec.api.records.SequenceRecord {

    private List<List<Writable>> sequenceRecord;
    private RecordMetaData metaData;

    @Override
    public int getSequenceLength() {
        if (sequenceRecord == null)
            return 0;
        return sequenceRecord.size();
    }

    @Override
    public List<Writable> getTimeStep(int timeStep) {
        if (timeStep < 0 || timeStep > sequenceRecord.size()) {
            throw new IllegalArgumentException("Invalid input: " + sequenceRecord.size()
                            + " time steps available; cannot get " + timeStep);
        }
        return sequenceRecord.get(timeStep);
    }
}
