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

package org.deeplearning4j.models.embeddings.inmemory;

import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class InMemoryLookupTableTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testConsumeOnEqualVocabs() throws Exception {
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        AbstractCache<VocabWord> cacheSource = new AbstractCache.Builder<VocabWord>().build();


        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");

        BasicLineIterator underlyingIterator = new BasicLineIterator(resource.getFile());


        SentenceTransformer transformer =
                        new SentenceTransformer.Builder().iterator(underlyingIterator).tokenizerFactory(t).build();

        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> vocabConstructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheSource).build();

        vocabConstructor.buildJointVocabulary(false, true);

        assertEquals(244, cacheSource.numWords());

        InMemoryLookupTable<VocabWord> mem1 =
                        (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>().vectorLength(100)
                                        .cache(cacheSource).seed(17).build();

        mem1.resetWeights(true);

        InMemoryLookupTable<VocabWord> mem2 =
                        (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>().vectorLength(100)
                                        .cache(cacheSource).seed(15).build();

        mem2.resetWeights(true);

        assertNotEquals(mem1.vector("day"), mem2.vector("day"));

        mem2.consume(mem1);

        assertEquals(mem1.vector("day"), mem2.vector("day"));

    }


    @Test
    public void testConsumeOnNonEqualVocabs() throws Exception {
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        AbstractCache<VocabWord> cacheSource = new AbstractCache.Builder<VocabWord>().build();


        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");

        BasicLineIterator underlyingIterator = new BasicLineIterator(resource.getFile());


        SentenceTransformer transformer =
                        new SentenceTransformer.Builder().iterator(underlyingIterator).tokenizerFactory(t).build();

        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> vocabConstructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheSource).build();

        vocabConstructor.buildJointVocabulary(false, true);

        assertEquals(244, cacheSource.numWords());

        InMemoryLookupTable<VocabWord> mem1 =
                        (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>().vectorLength(100)
                                        .cache(cacheSource).build();

        mem1.resetWeights(true);



        AbstractCache<VocabWord> cacheTarget = new AbstractCache.Builder<VocabWord>().build();



        FileLabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                        .addSourceFolder(new ClassPathResource("/paravec/labeled").getFile()).build();

        transformer = new SentenceTransformer.Builder().iterator(labelAwareIterator).tokenizerFactory(t).build();

        sequenceIterator = new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> vocabTransfer = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheTarget).build();

        vocabTransfer.buildMergedVocabulary(cacheSource, true);

        // those +3 go for 3 additional entries in target VocabCache: labels
        assertEquals(cacheSource.numWords() + 3, cacheTarget.numWords());


        InMemoryLookupTable<VocabWord> mem2 =
                        (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>().vectorLength(100)
                                        .cache(cacheTarget).seed(18).build();

        mem2.resetWeights(true);

        assertNotEquals(mem1.vector("day"), mem2.vector("day"));

        mem2.consume(mem1);

        assertEquals(mem1.vector("day"), mem2.vector("day"));

        assertTrue(mem1.syn0.rows() < mem2.syn0.rows());

        assertEquals(mem1.syn0.rows() + 3, mem2.syn0.rows());
    }
}
