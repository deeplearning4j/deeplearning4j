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

package org.datavec.local.transforms.functions.data;


import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.BytesWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.List;

/**SequenceRecordReaderBytesFunction: Converts binary data (in the form of a BytesWritable) to DataVec format data
 * ({@code List<List<<Writable>>}) using a SequenceRecordReader
 * @author Alex Black
 */
public class SequenceRecordReaderBytesFunction implements Function<Pair<Text, BytesWritable>, List<List<Writable>>> {

    private final SequenceRecordReader recordReader;

    public SequenceRecordReaderBytesFunction(SequenceRecordReader recordReader) {
        this.recordReader = recordReader;
    }

    @Override
    public List<List<Writable>> apply(Pair<Text, BytesWritable> v1) {
        URI uri = URI.create(v1.getFirst().toString());
        DataInputStream dis = new DataInputStream(new ByteArrayInputStream(v1.getRight().getContent()));
        try {
            return recordReader.sequenceRecord(uri, dis);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }

    }
}
