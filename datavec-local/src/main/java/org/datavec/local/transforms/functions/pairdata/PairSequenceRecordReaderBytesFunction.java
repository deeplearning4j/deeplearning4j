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

package org.datavec.local.transforms.functions.pairdata;


import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.List;

/**
 * SequenceRecordReaderBytesFunction: Converts two sets of binary data (in the form of a BytesPairWritable) to DataVec format data
 * ({@code Tuple2<List<List<<Writable>>,List<List<Writable>>}) using two SequenceRecordReaders.
 * Used for example when network input and output data comes from different files
 *
 * @author Alex Black
 */
public class PairSequenceRecordReaderBytesFunction implements
        Function<Pair<Text, BytesPairWritable>, Pair<List<List<Writable>>, List<List<Writable>>>> {
    private final SequenceRecordReader recordReaderFirst;
    private final SequenceRecordReader recordReaderSecond;

    public PairSequenceRecordReaderBytesFunction(SequenceRecordReader recordReaderFirst,
                    SequenceRecordReader recordReaderSecond) {
        this.recordReaderFirst = recordReaderFirst;
        this.recordReaderSecond = recordReaderSecond;
    }

    @Override
    public Pair<List<List<Writable>>, List<List<Writable>>> apply(Pair<Text, BytesPairWritable> v1) {
        BytesPairWritable bpw = v1.getRight();
        DataInputStream dis1 = new DataInputStream(new ByteArrayInputStream(bpw.getFirst()));
        DataInputStream dis2 = new DataInputStream(new ByteArrayInputStream(bpw.getSecond()));
        URI u1 = (bpw.getUriFirst() != null ? URI.create(bpw.getUriFirst()) : null);
        URI u2 = (bpw.getUriSecond() != null ? URI.create(bpw.getUriSecond()) : null);
        List<List<Writable>> first = null;
        try {
            first = recordReaderFirst.sequenceRecord(u1, dis1);
        } catch (IOException e) {
            e.printStackTrace();
        }
        List<List<Writable>> second = null;
        try {
            second = recordReaderSecond.sequenceRecord(u2, dis2);
        } catch (IOException e) {
            e.printStackTrace();
        }


        return Pair.of(first, second);
    }
}
