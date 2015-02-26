/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.cli.files;

import org.apache.commons.io.FilenameUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.factory.RecordReaderFactory;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;

import java.io.File;
import java.io.IOException;
import java.net.URI;

/**
 * @author sonali
 */
public class FileRecordReaderFactory implements RecordReaderFactory {
    @Override
    public RecordReader create(URI uri) throws Exception {
        File file = new File(uri.toString());
        InputSplit split = new FileSplit(file);

        String fileNameExtension = FilenameUtils.getExtension(uri.toString());

        switch (fileNameExtension) {
            case "csv":
                RecordReader recordReader = new CSVRecordReader();
                return recordReader;
            case "txt":
                RecordReader recordReader1 = new FileRecordReader();
                return recordReader1;
        }

    }

}
