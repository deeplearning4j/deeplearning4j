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

package org.datavec.api.split.partition;

import org.datavec.api.conf.Configuration;
import org.datavec.api.split.InputSplit;

import java.io.OutputStream;

/**
 * A partitioner for iterating thorugh files for {@link org.datavec.api.records.writer.RecordWriter}.
 * This allows for a configurable log rotation like algorithm for partitioning files by number of recodrds,
 * sizes among other things.
 */
public interface Partitioner {

    /**
     * Returns the total records written
     * @return
     */
    int totalRecordsWritten();

    /**
     * Number of records written so far
     *
     * @return
     */
    int numRecordsWritten();

    /**
     * Returns the number of partitions
     * @return
     */
    int numPartitions();

    /**
     * Initializes this partitioner with the given configuration
     * and input split
     * @param inputSplit the input split to use with this partitioner
     */
    void init(InputSplit inputSplit);

    /**
     * Initializes this partitioner with the given configuration
     * and input split
     * @param configuration the configuration to configure
     *                      this partitioner with
     * @param split the input split to use with this partitioner
     */
    void init(Configuration configuration,InputSplit split);

    /**
     * Updates the metadata for this partitioner
     * (to indicate whether the next partition is needed or not)
     * @param metadata
     */
    void updatePartitionInfo(PartitionMetaData metadata);

    /**
     * Returns true if the partition needs to be moved to the next.
     * This is controlled with {@link #updatePartitionInfo(PartitionMetaData)}
     * which handles incrementing counters and the like
     * to determine whether the current partition has been exhausted.
     * @return
     */
    boolean needsNewPartition();


    /**
     * "Increment" to the next stream
     * @return the new opened output stream
     */
    OutputStream openNewStream();

    /**
     * Get the current output stream
     * @return
     */
    OutputStream currentOutputStream();



}
