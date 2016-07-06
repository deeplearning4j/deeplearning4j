/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.api.records.writer.impl;



import org.canova.api.conf.Configuration;
import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;

/**
 * Line record writer
 * @author Adam Gibson
 */
public class LineRecordWriter extends FileRecordWriter {
    public LineRecordWriter() {
    }

    public LineRecordWriter(File path) throws FileNotFoundException {
        super(path);
    }

    public LineRecordWriter(File path, boolean append) throws FileNotFoundException {
        super(path, append);
    }

    public LineRecordWriter(Configuration conf) throws FileNotFoundException {
        super(conf);
    }

    @Override
    public void write(Collection<Writable> record) throws IOException {
         if(!record.isEmpty()) {
             Text t = (Text) record.iterator().next();
             t.write(out);
             out.write(NEW_LINE.getBytes());
         }


    }
}
