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

package org.datavec.local.transforms.functions;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.List;

/**RecordReaderFunction: Given a RecordReader and a file (via InputStream), load and parse the
 * data into a Collection<Writable>.
 * NOTE: This is only useful for "one record per file" type situations (ImageRecordReader, etc)
 * @author Alex Black
 */
public class RecordReaderFunction implements Function<Pair<String, InputStream>, List<Writable>> {
    protected RecordReader recordReader;

    public RecordReaderFunction(RecordReader recordReader) {
        this.recordReader = recordReader;
    }

    @Override
    public List<Writable> apply(Pair<String, InputStream> value) {
        URI uri = URI.create(value.getFirst());
        InputStream ds = value.getRight();
        try (DataInputStream dis = (DataInputStream) ds) {
            return recordReader.record(uri, dis);
        } catch (IOException e) {
            throw new IllegalStateException("Something went wrong reading file");
        }

    }
}
