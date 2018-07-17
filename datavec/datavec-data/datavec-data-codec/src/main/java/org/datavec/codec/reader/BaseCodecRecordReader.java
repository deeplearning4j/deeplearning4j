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

package org.datavec.codec.reader;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Codec record reader for parsing videos
 *
 * @author Adam Gibson
 */
public abstract class BaseCodecRecordReader extends FileRecordReader implements SequenceRecordReader {
    protected int startFrame = 0;
    protected int numFrames = -1;
    protected int totalFrames = -1;
    protected double framesPerSecond = -1;
    protected double videoLength = -1;
    protected int rows = 28, cols = 28;
    protected boolean ravel = false;

    public final static String NAME_SPACE = "org.datavec.codec.reader";
    public final static String ROWS = NAME_SPACE + ".rows";
    public final static String COLUMNS = NAME_SPACE + ".columns";
    public final static String START_FRAME = NAME_SPACE + ".startframe";
    public final static String TOTAL_FRAMES = NAME_SPACE + ".frames";
    public final static String TIME_SLICE = NAME_SPACE + ".time";
    public final static String RAVEL = NAME_SPACE + ".ravel";
    public final static String VIDEO_DURATION = NAME_SPACE + ".duration";


    @Override
    public List<List<Writable>> sequenceRecord() {
        if (iter == null || !iter.hasNext()) {
            this.advanceToNextLocation();
        }
        File next = iter.next();

        try {
            return loadData(next, null);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        return loadData(null, dataInputStream);
    }

    protected abstract List<List<Writable>> loadData(File file, InputStream inputStream) throws IOException;


    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        setConf(conf);
        initialize(split);
    }

    @Override
    public List<Writable> next() {
        throw new UnsupportedOperationException("next() not supported for CodecRecordReader (use: sequenceRecord)");
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("record(URI,DataInputStream) not supported for CodecRecordReader");
    }

    @Override
    public void setConf(Configuration conf) {
        super.setConf(conf);
        startFrame = conf.getInt(START_FRAME, 0);
        numFrames = conf.getInt(TOTAL_FRAMES, -1);
        rows = conf.getInt(ROWS, 28);
        cols = conf.getInt(COLUMNS, 28);
        framesPerSecond = conf.getFloat(TIME_SLICE, -1);
        videoLength = conf.getFloat(VIDEO_DURATION, -1);
        ravel = conf.getBoolean(RAVEL, false);
        totalFrames = conf.getInt(TOTAL_FRAMES, -1);
    }

    @Override
    public Configuration getConf() {
        return super.getConf();
    }

    @Override
    public SequenceRecord nextSequence() {
        File next = this.nextFile();

        List<List<Writable>> list;
        try {
            list = loadData(next, null);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return new org.datavec.api.records.impl.SequenceRecord(list,
                        new RecordMetaDataURI(next.toURI(), CodecRecordReader.class));
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadSequenceFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<SequenceRecord> out = new ArrayList<>();
        for (RecordMetaData meta : recordMetaDatas) {
            File f = new File(meta.getURI());

            List<List<Writable>> list = loadData(f, null);
            out.add(new org.datavec.api.records.impl.SequenceRecord(list, meta));
        }

        return out;
    }


}
