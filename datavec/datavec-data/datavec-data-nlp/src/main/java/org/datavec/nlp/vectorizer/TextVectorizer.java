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
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.vector.Vectorizer;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.nlp.metadata.DefaultVocabCache;
import org.datavec.nlp.metadata.VocabCache;
import org.datavec.nlp.stopwords.StopWords;
import org.datavec.nlp.tokenization.tokenizer.Tokenizer;
import org.datavec.nlp.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collection;

/**
 * Baseline text vectorizer that includes some common elements
 * to text analysis such as the tokenizer factory
 *
 * @author Adam Gibson
 */
public abstract class TextVectorizer<VECTOR_TYPE> implements Vectorizer<VECTOR_TYPE> {

    protected TokenizerFactory tokenizerFactory;
    protected int minWordFrequency = 0;
    public final static String MIN_WORD_FREQUENCY = "org.nd4j.nlp.minwordfrequency";
    public final static String STOP_WORDS = "org.nd4j.nlp.stopwords";
    public final static String TOKENIZER = "org.datavec.nlp.tokenizerfactory";
    public final static String VOCAB_CACHE = "org.datavec.nlp.vocabcache";
    protected Collection<String> stopWords;
    protected VocabCache cache;

    @Override
    public void initialize(Configuration conf) {
        tokenizerFactory = createTokenizerFactory(conf);
        minWordFrequency = conf.getInt(MIN_WORD_FREQUENCY, 5);
        stopWords = conf.getStringCollection(STOP_WORDS);
        if (stopWords == null || stopWords.isEmpty())
            stopWords = StopWords.getStopWords();

        String clazz = conf.get(VOCAB_CACHE, DefaultVocabCache.class.getName());
        try {
            Class<? extends VocabCache> tokenizerFactoryClazz = (Class<? extends VocabCache>) Class.forName(clazz);
            cache = tokenizerFactoryClazz.newInstance();
            cache.initialize(conf);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void fit(RecordReader reader) {
        fit(reader, null);
    }

    @Override
    public void fit(RecordReader reader, RecordCallBack callBack) {
        while (reader.hasNext()) {
            Record record = reader.nextRecord();
            String s = toString(record.getRecord());
            Tokenizer tokenizer = tokenizerFactory.create(s);
            cache.incrementNumDocs(1);
            doWithTokens(tokenizer);
            if (callBack != null)
                callBack.onRecord(record);


        }
    }


    protected Counter<String> wordFrequenciesForRecord(Collection<Writable> record) {
        String s = toString(record);
        Tokenizer tokenizer = tokenizerFactory.create(s);
        Counter<String> ret = new Counter<>();
        while (tokenizer.hasMoreTokens())
            ret.incrementCount(tokenizer.nextToken(), 1.0);
        return ret;
    }


    protected String toString(Collection<Writable> record) {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        for (Writable w : record) {
            if (w instanceof Text) {
                try {
                    w.write(dos);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        return new String(bos.toByteArray());
    }


    /**
     * Increment counts, add to collection,...
     * @param tokenizer
     */
    public abstract void doWithTokens(Tokenizer tokenizer);

    /**
     * Create tokenizer factory based on the configuration
     * @param conf the configuration to use
     * @return the tokenizer factory based on the configuration
     */
    public abstract TokenizerFactory createTokenizerFactory(Configuration conf);

}
