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

package org.datavec.api.records.writer.impl.csv;


import org.datavec.api.records.writer.impl.FileRecordWriter;
import org.datavec.api.split.partition.PartitionMetaData;
import org.datavec.api.writable.Writable;

import java.io.IOException;
import java.util.List;

/**
 * Csv record writer
 *
 * @author Adam Gibson
 */
public class CSVRecordWriter extends FileRecordWriter {
    public static final String DEFAULT_DELIMITER = ",";

    private final byte[] delimBytes;
    private boolean firstLine = true;

    public CSVRecordWriter() {
        delimBytes = DEFAULT_DELIMITER.getBytes(encoding);
    }


    @Override
    public boolean supportsBatch() {
        return true;
    }

    @Override
    public PartitionMetaData writeBatch(List<List<Writable>> batch) throws IOException {
        for(List<Writable> record : batch) {
            if (!record.isEmpty()) {
                //Add new line before appending lines rather than after (avoids newline after last line)
                if (!firstLine) {
                    out.write(NEW_LINE.getBytes());
                } else {
                    firstLine = false;
                }

                int count = 0;
                int last = record.size() - 1;
                for (Writable w : record) {
                    out.write(w.toString().getBytes(encoding));
                    if (count++ != last)
                        out.write(delimBytes);
                }

                out.flush();
            }
        }

        return PartitionMetaData.builder().numRecordsUpdated(batch.size()).build();
    }

    @Override
    public PartitionMetaData write(List<Writable> record) throws IOException {
        if (!record.isEmpty()) {
            //Add new line before appending lines rather than after (avoids newline after last line)
            if (!firstLine) {
                out.write(NEW_LINE.getBytes());
            } else {
                firstLine = false;
            }

            int count = 0;
            int last = record.size() - 1;
            for (Writable w : record) {
                out.write(w.toString().getBytes(encoding));
                if (count++ != last)
                    out.write(delimBytes);
            }

            out.flush();
        }

        return PartitionMetaData.builder().numRecordsUpdated(1).build();
    }
}
