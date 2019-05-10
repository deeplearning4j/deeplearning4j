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

package org.datavec.api.records.listener;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.writer.RecordWriter;

import java.io.Serializable;

/**
 * Each time a record is read or written, mainly used for debugging or visualization.
 *
 * @author saudet
 */
public interface RecordListener extends Serializable {
    /**
     * Get if listener invoked.
     */
    boolean invoked();

    /**
     * Change invoke to true.
     */
    void invoke();

    /**
     * Event listener for each record to be read.
     * @param reader the record reader
     * @param record in raw format (Collection, File, String, Writable, etc)
     */
    void recordRead(RecordReader reader, Object record);

    /**
     * Event listener for each record to be written.
     * @param writer the record writer
     * @param record in raw format (Collection, File, String, Writable, etc)
     */
    void recordWrite(RecordWriter writer, Object record);
}
