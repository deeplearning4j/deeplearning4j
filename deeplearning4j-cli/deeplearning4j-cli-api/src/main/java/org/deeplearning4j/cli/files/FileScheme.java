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

import org.canova.api.exceptions.UnknownFormatException;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.factory.RecordReaderFactory;
import org.canova.api.records.reader.factory.RecordWriterFactory;
import org.canova.api.records.writer.RecordWriter;
import org.deeplearning4j.cli.api.schemes.BaseScheme;

import java.net.URI;

/**
 * Creates record reader readerFactory for local file scheme
 * @author sonali
 */
public class FileScheme extends BaseScheme {

    /**
     * Process data input; create record reader
     * @param uri
     */
    @Override
    public RecordReader createReader(URI uri) {
        try {
          return readerFactory.create(uri);
        } catch (UnknownFormatException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Process data output; create record writer
     * @param uri destination for saving model
     */
    @Override
    public RecordWriter createWriter(URI uri) {
        try {
          return recordWriterFactory.create(uri);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }


    @Override
    public RecordReaderFactory createReaderFactory() {
        return new FileRecordReaderFactory();
    }

    @Override
    public RecordWriterFactory createWriterFactory() {
        return new FileRecordWriterFactory();
    }
}
