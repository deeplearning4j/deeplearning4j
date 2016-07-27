/*
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

package org.datavec.api.records.reader.impl.collection;


import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.conf.Configuration;
import org.datavec.api.writable.Writable;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

/**
 * Collection record reader.
 * Mainly used for testing.
 *
 * @author Adam Gibson
 */
public class CollectionRecordReader extends BaseRecordReader {
    private Iterator<? extends Collection<Writable>> records;
    private final Collection<? extends Collection<Writable>> original;

    public CollectionRecordReader(Collection<? extends Collection<Writable>> records) {
        this.records = records.iterator();
        this.original = records;
    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {

    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        initialize(split);
    }

    @Override
    public List<Writable> next() {
        Collection<Writable> next = records.next();
        List<Writable> record = (next instanceof List ? (List<Writable>)next : new ArrayList<>(next));
        invokeListeners(record);
        return record;
    }

    @Override
    public boolean hasNext() {
        return records.hasNext();
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {

    }

    @Override
    public Configuration getConf() {
        return null;
    }

    @Override
    public List<String> getLabels(){
        return null;
    }

    @Override
    public void reset() {
        this.records = original.iterator();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException("Generating records from DataInputStream not supported for CollectionRecordReader");
    }


}
