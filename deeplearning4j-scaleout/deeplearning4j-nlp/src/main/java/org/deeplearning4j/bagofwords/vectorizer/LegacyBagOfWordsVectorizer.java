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

import org.apache.commons.io.IOUtils;
import org.apache.uima.util.FileUtils;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.InputStream;
import java.util.List;

/**
 * Bag of words vectorizer.
 * Transforms a document in to a bag of words
 * @author Adam Gibson
 *
 */
@Deprecated
public class LegacyBagOfWordsVectorizer extends LegacyBaseTextVectorizer {


    public LegacyBagOfWordsVectorizer(){}

    protected LegacyBagOfWordsVectorizer(VocabCache cache,
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
        super(cache, tokenizerFactory, stopWords, minWordFrequency, docIter, sentenceIterator,
            labels,index,batchSize,sample,stem,cleanup);
    }

    /**
     * Text coming from an input stream considered as one document
     *
     * @param is    the input stream to read from
     * @param label the label to assign
     * @return a dataset with a transform of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(InputStream is, String label) {
        try {
            String inputString = IOUtils.toString(is);
            return vectorize(inputString,label);

        }catch(Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Vectorizes the passed in text treating it as one document
     *
     * @param text  the text to vectorize
     * @param label the label of the text
     * @return a dataset with a transform of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(String text, String label) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray input = Nd4j.create(1,cache.numWords());
        for (String token : tokens) {
          int idx = cache.indexOf(token);
          if (cache.indexOf(token) >= 0)
            input.putScalar(idx, cache.wordFrequency(token));
        }

        INDArray labelMatrix = FeatureUtil.toOutcomeVector(labels.indexOf(label), labels.size());
        return new DataSet(input,labelMatrix);
    }


    /**
     * @param input the text to vectorize
     * @param label the label of the text
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(File input, String label) {
        try {
            String text = FileUtils.file2String(input);
            return vectorize(text, label);
        }catch(Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Transforms the matrix
     *
     * @param text
     * @return
     */
    @Override
    public INDArray transform(String text) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray input = Nd4j.create(1, cache.numWords());
        for (String token : tokens) {
         int idx = cache.indexOf(token);
         if (cache.indexOf(token) >= 0)
          input.putScalar(idx, cache.wordFrequency(token));
        }
        return input;
    }

    @Override
    public DataSet vectorize() {
        return null;
    }


    public static class Builder extends org.deeplearning4j.bagofwords.vectorizer.Builder {

        @Override
        public TextVectorizer build() {
            return new LegacyBagOfWordsVectorizer(cache, tokenizerFactory, stopWords, minWordFrequency,
                docIter, sentenceIterator,labels,index,batchSize,sample,stem,cleanup);
        }
    }

}
