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

package org.canova.api.formats.output.impl;


import org.canova.api.conf.Configuration;
import org.canova.api.exceptions.CanovaException;
import org.canova.api.formats.output.OutputFormat;
import org.canova.api.records.writer.RecordWriter;
import org.canova.api.records.writer.impl.CSVRecordWriter;

import java.io.File;
import java.io.FileNotFoundException;

/**
 * Creates an @link{CSVRecordWriter}
 *
 * @author Adam Gibson
 */
public class CSVOutputFormat implements OutputFormat {
    @Override
    public RecordWriter createWriter(Configuration conf) throws CanovaException {
        String outputPath = conf.get(OutputFormat.OUTPUT_PATH,".");
        try {
            return new CSVRecordWriter(new File(outputPath));
        } catch (FileNotFoundException e) {
            throw new CanovaException(e);
        }
    }
}
