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
import org.datavec.api.records.writer.RecordWriter;

import java.io.IOException;

/**
 * Created by Alex on 07/07/2017.
 */
public class RecordReaderConverter {

    private RecordReaderConverter() { }

    public static void convert(RecordReader reader, RecordWriter writer) throws IOException {
        convert(reader, writer, true);
    }

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

}
