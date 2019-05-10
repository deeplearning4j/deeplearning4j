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

package org.deeplearning4j.bagofwords.vectorizer;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class TfidfVectorizer extends BaseTextVectorizer {
    /**
     * Text coming from an input stream considered as one document
     *
     * @param is    the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(InputStream is, String label) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            String line = "";
            StringBuilder builder = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                builder.append(line);
            }
            return vectorize(builder.toString(), label);
        } catch (Exception e) {
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
        INDArray input = transform(text);
        INDArray labelMatrix = FeatureUtil.toOutcomeVector(labelsSource.indexOf(label), labelsSource.size());

        return new DataSet(input, labelMatrix);
    }

    /**
     * @param input the text to vectorize
     * @param label the label of the text
     * @return {@link DataSet} with a applyTransformToDestination of
     * weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(File input, String label) {
        try {
            String string = FileUtils.readFileToString(input);
            return vectorize(string, label);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Transforms the matrix
     *
     * @param text text to transform
     * @return {@link INDArray}
     */
    @Override
    public INDArray transform(String text) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();

        // build document words count
        return transform(tokens);
    }


    @Override
    public INDArray transform(List<String> tokens) {
        INDArray ret = Nd4j.create(1, vocabCache.numWords());

        Map<String, AtomicLong> counts = new HashMap<>();
        for (String token : tokens) {
            if (!counts.containsKey(token))
                counts.put(token, new AtomicLong(0));

            counts.get(token).incrementAndGet();
        }

        for (int i = 0; i < tokens.size(); i++) {
            int idx = vocabCache.indexOf(tokens.get(i));
            if (idx >= 0) {
                double tf_idf = tfidfWord(tokens.get(i), counts.get(tokens.get(i)).longValue(), tokens.size());
                //log.info("TF-IDF for word: {} -> {} / {} => {}", tokens.get(i), counts.get(tokens.get(i)).longValue(), tokens.size(), tf_idf);
                ret.putScalar(idx, tf_idf);
            }
        }
        return ret;
    }

    public double tfidfWord(String word, long wordCount, long documentLength) {
        //log.info("word: {}; TF: {}; IDF: {}", word, tfForWord(wordCount, documentLength), idfForWord(word));
        return MathUtils.tfidf(tfForWord(wordCount, documentLength), idfForWord(word));
    }

    private double tfForWord(long wordCount, long documentLength) {
        return (double) wordCount / (double) documentLength;
    }

    private double idfForWord(String word) {
        return MathUtils.idf(vocabCache.totalNumberOfDocs(), vocabCache.docAppearedIn(word));
    }


    /**
     * Vectorizes the input source in to a dataset
     *
     * @return Adam Gibson
     */
    @Override
    public DataSet vectorize() {
        return null;
    }

    public static class Builder {
        protected TokenizerFactory tokenizerFactory;
        protected LabelAwareIterator iterator;
        protected int minWordFrequency;
        protected VocabCache<VocabWord> vocabCache;
        protected LabelsSource labelsSource = new LabelsSource();
        protected Collection<String> stopWords = new ArrayList<>();
        protected boolean isParallel = true;

        public Builder() {}

        public Builder allowParallelTokenization(boolean reallyAllow) {
            this.isParallel = reallyAllow;
            return this;
        }

        public Builder setTokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder setIterator(@NonNull LabelAwareIterator iterator) {
            this.iterator = iterator;
            return this;
        }

        public Builder setIterator(@NonNull DocumentIterator iterator) {
            this.iterator = new DocumentIteratorConverter(iterator, labelsSource);
            return this;
        }

        public Builder setIterator(@NonNull SentenceIterator iterator) {
            this.iterator = new SentenceIteratorConverter(iterator, labelsSource);
            return this;
        }

        public Builder setVocab(@NonNull VocabCache<VocabWord> vocab) {
            this.vocabCache = vocab;
            return this;
        }

        public Builder setMinWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder setStopWords(Collection<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public TfidfVectorizer build() {
            TfidfVectorizer vectorizer = new TfidfVectorizer();

            vectorizer.tokenizerFactory = this.tokenizerFactory;
            vectorizer.iterator = this.iterator;
            vectorizer.minWordFrequency = this.minWordFrequency;
            vectorizer.labelsSource = this.labelsSource;
            vectorizer.isParallel = this.isParallel;

            if (this.vocabCache == null) {
                this.vocabCache = new AbstractCache.Builder<VocabWord>().build();
            }

            vectorizer.vocabCache = this.vocabCache;
            vectorizer.stopWords = this.stopWords;

            return vectorizer;
        }

    }
}
