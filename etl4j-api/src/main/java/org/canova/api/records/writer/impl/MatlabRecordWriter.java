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
import org.canova.api.writable.Writable;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Collection;

/**
 * Write matlab records
 *
 * @author Adam Gibson
 */
public class MatlabRecordWriter extends FileRecordWriter {
    public MatlabRecordWriter() {
    }

    public MatlabRecordWriter(File path) throws FileNotFoundException {
        super(path);
    }

    public MatlabRecordWriter(File path, boolean append) throws FileNotFoundException {
        super(path, append);
    }

    public MatlabRecordWriter(Configuration conf) throws FileNotFoundException {
        super(conf);
    }

    @Override
    public void write(Collection<Writable> record) throws IOException {
        StringBuilder result = new StringBuilder();

        int count = 0;
        for(Writable w : record) {
            // attributes
            if (count > 0) {
              boolean tabs = false;
              result.append((tabs ? "\t" : " "));
            }
            result.append(w.toString());
            count++;

        }

        out.write(result.toString().getBytes());
        out.write(NEW_LINE.getBytes());



    }
}
