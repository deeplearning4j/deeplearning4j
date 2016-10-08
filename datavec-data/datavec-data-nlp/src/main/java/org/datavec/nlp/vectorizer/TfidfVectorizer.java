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

package org.datavec.nlp.vectorizer;


import org.datavec.api.berkeley.Counter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.nlp.reader.TfidfRecordReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 *
 * Nd4j tfidf vectorizer
 *
 * @author Adam Gibson
 */
public class TfidfVectorizer extends AbstractTfidfVectorizer<INDArray> {
    @Override
    public INDArray createVector(Object[] args) {
        Counter<String> docFrequencies = (Counter<String>)args[0];
        double[] vector = new double[cache.vocabWords().size()];
        for(int i = 0; i < cache.vocabWords().size(); i++) {
            double freq = docFrequencies.getCount(cache.wordAt(i));
            vector[i] = cache.tfidf(cache.wordAt(i),freq);
        }
        return Nd4j.create(vector);
    }

    @Override
    public INDArray fitTransform(RecordReader reader) {
        return fitTransform(reader,null);
    }

    @Override
    public INDArray fitTransform(final RecordReader reader, RecordCallBack callBack) {
        final List<Collection<Writable>> records = new ArrayList<>();
        final TfidfRecordReader reader2 = (TfidfRecordReader) reader;
        fit(reader,new RecordCallBack() {
            @Override
            public void onRecord(Collection<Writable> record) {
                if(reader.getConf().get(TfidfRecordReader.APPEND_LABEL).equals("true")) {
                    record.add(new IntWritable(reader2.getCurrentLabel()));
                }
                records.add(record);
            }
        });

        if(records.isEmpty())
            throw new IllegalStateException("No records found!");
        INDArray ret = Nd4j.create(records.size(),cache.vocabWords().size());
        int i = 0;
        for(Collection<Writable> record : records) {
            ret.putRow(i++, transform(record));
            if(callBack != null) {
                callBack.onRecord(record);
            }
        }

        return ret;
    }

    @Override
    public INDArray transform(Collection<Writable> record) {
        Counter<String> wordFrequencies = wordFrequenciesForRecord(record);
        return createVector(new Object[]{wordFrequencies});

    }
}
