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

package org.datavec.nlp.vectorizer;


import org.nd4j.linalg.primitives.Counter;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.NDArrayWritable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
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
        Counter<String> docFrequencies = (Counter<String>) args[0];
        double[] vector = new double[cache.vocabWords().size()];
        for (int i = 0; i < cache.vocabWords().size(); i++) {
            double freq = docFrequencies.getCount(cache.wordAt(i));
            vector[i] = cache.tfidf(cache.wordAt(i), freq);
        }
        return Nd4j.create(vector);
    }

    @Override
    public INDArray fitTransform(RecordReader reader) {
        return fitTransform(reader, null);
    }

    @Override
    public INDArray fitTransform(final RecordReader reader, RecordCallBack callBack) {
        final List<Record> records = new ArrayList<>();
        fit(reader, new RecordCallBack() {
            @Override
            public void onRecord(Record record) {
                records.add(record);
            }
        });

        if (records.isEmpty())
            throw new IllegalStateException("No records found!");
        INDArray ret = Nd4j.create(records.size(), cache.vocabWords().size());
        int i = 0;
        for (Record record : records) {
            INDArray transformed = transform(record);
            org.datavec.api.records.impl.Record transformedRecord = new org.datavec.api.records.impl.Record(
                            Arrays.asList(new NDArrayWritable(transformed),
                                            record.getRecord().get(record.getRecord().size() - 1)),
                            new RecordMetaDataURI(record.getMetaData().getURI(), reader.getClass()));
            ret.putRow(i++, transformed);
            if (callBack != null) {
                callBack.onRecord(transformedRecord);
            }
        }

        return ret;
    }

    @Override
    public INDArray transform(Record record) {
        Counter<String> wordFrequencies = wordFrequenciesForRecord(record.getRecord());
        return createVector(new Object[] {wordFrequencies});
    }
}
