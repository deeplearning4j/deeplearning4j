/*
 *
 *  *
 *  *  * Copyright 2015 Skymind,Inc.
 *  *  *
 *  *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *  *    you may not use this file except in compliance with the License.
 *  *  *    You may obtain a copy of the License at
 *  *  *
 *  *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *  *
 *  *  *    Unless required by applicable law or agreed to in writing, software
 *  *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *  *    See the License for the specific language governing permissions and
 *  *  *    limitations under the License.
 *  *
 *
 */

package org.canova.nlp.vectorizer;

import org.canova.api.berkeley.Counter;
import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.vector.Vectorizer;
import org.canova.api.writable.Writable;
import org.canova.nlp.metadata.DefaultVocabCache;
import org.canova.nlp.metadata.VocabCache;
import org.canova.nlp.stopwords.StopWords;
import org.canova.nlp.tokenization.tokenizer.Tokenizer;
import org.canova.nlp.tokenization.tokenizerfactory.TokenizerFactory;

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
    public final static String TOKENIZER = "org.canova.nlp.tokenizerfactory";
    protected Collection<String> stopWords;
    protected VocabCache cache;

    @Override
    public void initialize(Configuration conf) {
        tokenizerFactory = createTokenizerFactory(conf);
        minWordFrequency = conf.getInt(MIN_WORD_FREQUENCY,5);
        stopWords = conf.getStringCollection(STOP_WORDS);
        if(stopWords == null || stopWords.isEmpty())
            stopWords = StopWords.getStopWords();
        cache = new DefaultVocabCache(minWordFrequency);

    }

    @Override
    public void fit(RecordReader reader) {
        fit(reader,null);
    }

    @Override
    public void fit(RecordReader reader, RecordCallBack callBack) {
        while(reader.hasNext()) {
            Collection<Writable> record = reader.next();
            String s = toString(record);
            Tokenizer tokenizer = tokenizerFactory.create(s);
            cache.incrementNumDocs(1);
            doWithTokens(tokenizer);
            if(callBack != null)
                callBack.onRecord(record);


        }
    }


    protected Counter<String> wordFrequenciesForRecord(Collection<Writable> record) {
        String s = toString(record);
        Tokenizer tokenizer = tokenizerFactory.create(s);
        Counter<String> ret = new Counter<>();
        while(tokenizer.hasMoreTokens())
            ret.incrementCount(tokenizer.nextToken(),1.0);
        return ret;
    }


    protected String toString(Collection<Writable> record) {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        for(Writable w : record) {
            try {
                w.write(dos);
            } catch (IOException e) {
                e.printStackTrace();
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
