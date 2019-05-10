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

package org.datavec.api.records;

import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.List;

/**
 * A Record contains a set of values for a single example or instance. Each value in the Record is represented by
 * a {@link Writable} object. The record may (optionally) also have a {@link RecordMetaData} instance, that represents
 * metadata (source location, etc) for the record.<br>
 * For sequences, see {@link SequenceRecord}
 *
 * @author Alex Black
 */
public interface Record extends Serializable {

    /**
     * Get the record values, as a {@code List<Writable>}
     *
     * @return Record values
     */
    List<Writable> getRecord();

    /**
     * Get the record values for this Record
     */
    void setRecord(List<Writable> record);

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
