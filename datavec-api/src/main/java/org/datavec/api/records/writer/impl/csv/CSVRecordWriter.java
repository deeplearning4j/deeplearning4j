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

package org.datavec.api.records.writer.impl.csv;


import org.datavec.api.records.writer.impl.FileRecordWriter;
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
    public void writeBatch(List<List<Writable>> batch) throws IOException {
        for(List<Writable> record : batch) {
            write(record);
        }
    }

    @Override
    public void write(List<Writable> record) throws IOException {
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
}
