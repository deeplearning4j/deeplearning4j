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


import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.List;

/**RecordReaderFunction: Given a SequenceRecordReader and a file(via InputStream), load and parse the
 * sequence data into a {@code List<List<Writable>>}
 * @author Alex Black
 */
public class SequenceRecordReaderFunction
                implements Function<Pair<String, InputStream>, List<List<Writable>>> {
    protected SequenceRecordReader sequenceRecordReader;

    public SequenceRecordReaderFunction(SequenceRecordReader sequenceRecordReader) {
        this.sequenceRecordReader = sequenceRecordReader;
    }

    @Override
    public List<List<Writable>> apply(Pair<String, InputStream> value) {
        URI uri = URI.create(value.getFirst());
        try (DataInputStream dis = (DataInputStream) value.getRight()) {
            return sequenceRecordReader.sequenceRecord(uri, dis);
        } catch (IOException e) {
            e.printStackTrace();
        }

        throw new IllegalStateException("Something went wrong");
    }
}
