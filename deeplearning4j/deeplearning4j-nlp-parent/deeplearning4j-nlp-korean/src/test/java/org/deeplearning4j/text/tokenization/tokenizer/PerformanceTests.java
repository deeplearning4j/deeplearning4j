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

package org.deeplearning4j.text.tokenization.tokenizer;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.KoreanTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class PerformanceTests {


    @Ignore
    @Test
    public void testWord2VecCBOWBig() throws Exception {
        SentenceIterator iter = new BasicLineIterator("/home/raver119/Downloads/corpus/namuwiki_raw.txt");
        //iter = new BasicLineIterator("/home/raver119/Downloads/corpus/ru_sentences.txt");
        //SentenceIterator iter = new BasicLineIterator("/ext/DATASETS/ru/Socials/ru_sentences.txt");

        TokenizerFactory t = new KoreanTokenizerFactory();
        //t = new DefaultTokenizerFactory();
        //t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(5).learningRate(0.025).layerSize(150)
                        .seed(42).sampling(0).negativeSample(0).useHierarchicSoftmax(true).windowSize(5)
                        .modelUtils(new BasicModelUtils<VocabWord>()).useAdaGrad(false).iterate(iter).workers(8)
                        .allowParallelTokenization(true).tokenizerFactory(t)
                        .elementsLearningAlgorithm(new CBOW<VocabWord>()).build();

        long time1 = System.currentTimeMillis();

        vec.fit();

        long time2 = System.currentTimeMillis();

        log.info("Total execution time: {}", (time2 - time1));
    }
}
