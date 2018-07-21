/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.records.reader.impl.jackson;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * The sequence record reader version of {@link JacksonLineRecordReader}.<br>
 * Assumptions here:<br>
 * 1. Each file is a separate record<br>
 * 2. Each line of a file is one step within a sequence<br>
 * See {@link JacksonLineRecordReader} for more details
 *
 *
 * @author Alex Black
 */
public class JacksonLineSequenceRecordReader extends FileRecordReader implements SequenceRecordReader {

    private FieldSelection selection;
    private ObjectMapper mapper;

    /**
     *
     * @param selection Fields to return
     * @param mapper    Object mapper to use. For example, {@code new ObjectMapper(new JsonFactory())} for JSON
     */
    public JacksonLineSequenceRecordReader(FieldSelection selection, ObjectMapper mapper){
        this.selection = selection;
        this.mapper = mapper;
    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        return nextSequence().getSequenceRecord();
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return loadAndClose(dataInputStream);
    }

    @Override
    public SequenceRecord nextSequence() {
        if(!hasNext()){
            throw new NoSuchElementException("No next element");
        }

        File next = iter.next();
        List<List<Writable>> out;
        try {
            out = loadAndClose(new FileInputStream(next));
        } catch (IOException e){
            throw new RuntimeException(e);
        }
        return new org.datavec.api.records.impl.SequenceRecord(out, new RecordMetaDataURI(next.toURI()));
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return null;
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<SequenceRecord> out = new ArrayList<>();
        for(RecordMetaData m : recordMetaDatas){
            out.add(loadSequenceFromMetaData(m));
        }
        return out;
    }

    private List<List<Writable>> loadAndClose(InputStream inputStream) {
        LineIterator lineIter = null;
        try {
            lineIter = IOUtils.lineIterator(new BufferedReader(new InputStreamReader(inputStream)));
            return load(lineIter);
        } finally {
            if (lineIter != null) {
                lineIter.close();
            }
            IOUtils.closeQuietly(inputStream);
        }
    }

    private List<List<Writable>> load(Iterator<String> lineIter) {
        List<List<Writable>> out = new ArrayList<>();
        while(lineIter.hasNext()){
            out.add(JacksonReaderUtils.parseRecord(lineIter.next(), selection, mapper));
        }
        return out;
    }
}
