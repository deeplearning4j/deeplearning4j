/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.bagofwords.vectorizer;


import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.List;

/**
 * Turns a applyTransformToDestination of documents in to a tfidf bag of words
 * @author Adam Gibson
 */
@Deprecated
public class LegacyTfidfVectorizer extends LegacyBaseTextVectorizer implements Serializable {

    public LegacyTfidfVectorizer(){}

    protected LegacyTfidfVectorizer(VocabCache cache,
                                    TokenizerFactory tokenizerFactory,
                                    List<String> stopWords,
                                    int minWordFrequency,
                                    DocumentIterator docIter,
                                    SentenceIterator sentenceIterator,
                                    List<String> labels,
                                    InvertedIndex index,
                                    int batchSize,
                                    double sample,
                                    boolean stem,
                                    boolean cleanup) {
        super(cache, tokenizerFactory, stopWords, minWordFrequency, docIter, sentenceIterator, labels,index,batchSize,sample,stem,cleanup);
    }

    private double tfidfWord(String word) {
        return MathUtils.tfidf(tfForWord(word),idfForWord(word));
    }


    private double tfForWord(String word) {
        //return MathUtils.tf(cache.wordFrequency(word));
        return 0;
    }

    private double idfForWord(String word) {
        return MathUtils.idf(cache.totalNumberOfDocs(),cache.docAppearedIn(word));
    }


    private INDArray tfidfForInput(String text) {
        INDArray ret = Nd4j.create(1, cache.numWords());
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();

        for(int i = 0;i < tokens.size(); i++) {
            int idx = cache.indexOf(tokens.get(i));
            if(idx >= 0)
                ret.putScalar(idx, tfidfWord(tokens.get(i)));
        }
        return ret;
    }

    private INDArray tfidfForInput(InputStream is) {
        try {
            String text = new String(IOUtils.toByteArray(is));
            return tfidfForInput(text);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public DataSet vectorize(InputStream is, String label) {
        return new DataSet(tfidfForInput(is),FeatureUtil.toOutcomeVector(labels.indexOf(label),labels.size()));
    }

    @Override
    public DataSet vectorize(String text, String label) {
        if (labels.indexOf(label) < 0) {
            throw new IllegalArgumentException("Label not found in a dictionary.");
        }
        INDArray tfidf  = tfidfForInput(text);
        INDArray label2 = FeatureUtil.toOutcomeVector(labels.indexOf(label), labels.size());
        return new DataSet(tfidf,label2);
    }



    @Override
    public DataSet vectorize(File input, String label) {
        try {
            return vectorize(FileUtils.readFileToString(input),label);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Transforms the matrix
     *
     * @param text
     * @return {@link INDArray}
     */
    @Override
    public INDArray transform(String text) {
        return tfidfForInput(text);
    }

    @Override
    public DataSet vectorize() {
        return null;
    }

    public static class Builder extends org.deeplearning4j.bagofwords.vectorizer.Builder {

        public TextVectorizer build() {

            return new LegacyTfidfVectorizer(cache, tokenizerFactory, stopWords, minWordFrequency, docIter, sentenceIterator,labels,index,batchSize,sample,stem,cleanup);

        }

    }


}
