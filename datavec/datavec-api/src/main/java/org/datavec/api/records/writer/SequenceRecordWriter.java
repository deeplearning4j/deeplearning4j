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

package org.datavec.api.records.writer;



import org.datavec.api.conf.Configurable;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.writable.Writable;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;

/**
 *  Sequence record writer
 *
 *  @author Alex Black
 */
public interface SequenceRecordWriter extends Closeable, Configurable {
    String APPEND = "org.datavec.api.record.writer.append";

    /**
     * Write a record
     * @param sequence the record to write
     */
    PartitionMetaData write(List<List<Writable>> sequence) throws IOException;


    /**
     * Close the sequence record writer
     */
    void close();

}
