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

package org.datavec.nlp.reader;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.vector.Vectorizer;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.nlp.vectorizer.TfidfVectorizer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.*;

/**
 * TFIDF record reader (wraps a tfidf vectorizer
 * for delivering labels and conforming to the record reader interface)
 *
 * @author Adam Gibson
 */
public class TfidfRecordReader extends FileRecordReader {
    private TfidfVectorizer tfidfVectorizer;
    private List<Record> records = new ArrayList<>();
    private Iterator<Record> recordIter;
    private int numFeatures;
    private boolean initialized = false;


    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        initialize(new Configuration(), split);
    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        //train  a new one since it hasn't been specified
        if (tfidfVectorizer == null) {
            tfidfVectorizer = new TfidfVectorizer();
            tfidfVectorizer.initialize(conf);

            //clear out old strings
            records.clear();

            INDArray ret = tfidfVectorizer.fitTransform(this, new Vectorizer.RecordCallBack() {
                @Override
                public void onRecord(Record fullRecord) {
                    records.add(fullRecord);
                }
            });

            //cache the number of features used for each document
            numFeatures = ret.columns();
            recordIter = records.iterator();
        } else {
            records = new ArrayList<>();

            //the record reader has 2 phases, we are skipping the
            //document frequency phase and just using the super() to get the file contents
            //and pass it to the already existing vectorizer.
            while (super.hasNext()) {
                Record fileContents = super.nextRecord();
                INDArray transform = tfidfVectorizer.transform(fileContents);

                org.datavec.api.records.impl.Record record = new org.datavec.api.records.impl.Record(
                                new ArrayList<>(Collections.<Writable>singletonList(new NDArrayWritable(transform))),
                                new RecordMetaDataURI(fileContents.getMetaData().getURI(), TfidfRecordReader.class));

                if (appendLabel)
                    record.getRecord().add(fileContents.getRecord().get(fileContents.getRecord().size() - 1));

                records.add(record);
            }

            recordIter = records.iterator();
        }

        this.initialized = true;
    }

    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        recordIter = records.iterator();
    }

    @Override
    public Record nextRecord() {
        if (recordIter == null)
            return super.nextRecord();
        return recordIter.next();
    }

    @Override
    public List<Writable> next() {
        return nextRecord().getRecord();
    }

    @Override
    public boolean hasNext() {
        //we aren't done vectorizing yet
        if (recordIter == null)
            return super.hasNext();
        return recordIter.hasNext();
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    public TfidfVectorizer getTfidfVectorizer() {
        return tfidfVectorizer;
    }

    public void setTfidfVectorizer(TfidfVectorizer tfidfVectorizer) {
        if (initialized) {
            throw new IllegalArgumentException(
                            "Setting TfidfVectorizer after TfidfRecordReader initialization doesn't have an effect");
        }
        this.tfidfVectorizer = tfidfVectorizer;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public void shuffle() {
        this.shuffle(new Random());
    }

    public void shuffle(Random random) {
        Collections.shuffle(this.records, random);
        this.reset();
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> out = new ArrayList<>();

        for (Record fileContents : super.loadFromMetaData(recordMetaDatas)) {
            INDArray transform = tfidfVectorizer.transform(fileContents);

            org.datavec.api.records.impl.Record record = new org.datavec.api.records.impl.Record(
                            new ArrayList<>(Collections.<Writable>singletonList(new NDArrayWritable(transform))),
                            new RecordMetaDataURI(fileContents.getMetaData().getURI(), TfidfRecordReader.class));

            if (appendLabel)
                record.getRecord().add(fileContents.getRecord().get(fileContents.getRecord().size() - 1));
            out.add(record);
        }

        return out;
    }
}

