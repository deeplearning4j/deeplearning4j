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

package org.deeplearning4j.cli.api.schemes;

import org.canova.api.records.reader.factory.RecordReaderFactory;
import org.canova.api.records.reader.factory.RecordWriterFactory;
import org.canova.api.writable.Writable;

import java.net.URI;
import java.util.Collection;

/**
 * Base scheme for reading in data via Canova RecordReader
 * @author sonali
 */
public abstract class BaseScheme implements Scheme {
    protected RecordReaderFactory readerFactory;
    protected RecordWriterFactory recordWriterFactory;

    public BaseScheme() {
        readerFactory = createReaderFactory();
        recordWriterFactory = createWriterFactory();
    }

    public abstract RecordReaderFactory createReaderFactory();
    public abstract RecordWriterFactory createWriterFactory();

    public RecordWriterFactory writerFactory() {
        return recordWriterFactory;
    }

    public RecordReaderFactory readerFactory() {
        return readerFactory;
    }
}
