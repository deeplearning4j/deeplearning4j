/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.records.converter;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.SequenceRecordWriter;

import java.io.IOException;

/**
 * A utility class to aid in the conversion of data from one {@link RecordReader} to one {@link RecordWriter},
 * or from one {@link SequenceRecordReader} to one {@link SequenceRecordWriter}
 *
 * @author Alex Black
 */
public class RecordReaderConverter {

    private RecordReaderConverter() { }

    /**
     * Write all values from the specified record reader to the specified record writer.
     * Closes the record writer on completion
     *
     * @param reader Record reader (source of data)
     * @param writer Record writer (location to write data)
     * @throws IOException If underlying reader/writer throws an exception
     */
    public static void convert(RecordReader reader, RecordWriter writer) throws IOException {
        convert(reader, writer, true);
    }

    /**
     * Write all values from the specified record reader to the specified record writer.
     * Optionally, close the record writer on completion
     *
     * @param reader Record reader (source of data)
     * @param writer Record writer (location to write data)
     * @param closeOnCompletion if true: close the record writer once complete, via {@link RecordWriter#close()}
     * @throws IOException If underlying reader/writer throws an exception
     */
    public static void convert(RecordReader reader, RecordWriter writer, boolean closeOnCompletion) throws IOException {

        if(!reader.hasNext()){
            throw new UnsupportedOperationException("Cannot convert RecordReader: reader has no next element");
        }

        while(reader.hasNext()){
            writer.write(reader.next());
        }

        if(closeOnCompletion){
            writer.close();
        }
    }

    /**
     * Write all sequences from the specified sequence record reader to the specified sequence record writer.
     * Closes the sequence record writer on completion.
     *
     * @param reader Sequence record reader (source of data)
     * @param writer Sequence record writer (location to write data)
     * @throws IOException If underlying reader/writer throws an exception
     */
    public static void convert(SequenceRecordReader reader, SequenceRecordWriter writer) throws IOException {
        convert(reader, writer, true);
    }

    /**
     * Write all sequences from the specified sequence record reader to the specified sequence record writer.
     * Closes the sequence record writer on completion.
     *
     * @param reader Sequence record reader (source of data)
     * @param writer Sequence record writer (location to write data)
     * @param closeOnCompletion if true: close the record writer once complete, via {@link SequenceRecordWriter#close()}
     * @throws IOException If underlying reader/writer throws an exception
     */
    public static void convert(SequenceRecordReader reader, SequenceRecordWriter writer, boolean closeOnCompletion) throws IOException {

        if(!reader.hasNext()){
            throw new UnsupportedOperationException("Cannot convert SequenceRecordReader: reader has no next element");
        }

        while(reader.hasNext()){
            writer.write(reader.sequenceRecord());
        }

        if(closeOnCompletion){
            writer.close();
        }
    }

}
