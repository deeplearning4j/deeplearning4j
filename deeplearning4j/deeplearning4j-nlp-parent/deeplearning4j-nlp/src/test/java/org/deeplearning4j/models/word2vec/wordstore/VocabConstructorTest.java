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

package org.deeplearning4j.models.word2vec.wordstore;

import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
public class VocabConstructorTest {

    protected static final Logger log = LoggerFactory.getLogger(VocabConstructorTest.class);

    TokenizerFactory t = new DefaultTokenizerFactory();


    @Before
    public void setUp() throws Exception {
        t.setTokenPreProcessor(new CommonPreprocessor());
    }

    @Test
    public void testVocab() throws Exception {
        File inputFile = new ClassPathResource("big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);

        Set<String> set = new HashSet<>();
        int lines = 0;
        int cnt = 0;
        while (iter.hasNext()) {
            Tokenizer tok = t.create(iter.nextSentence());
            for (String token : tok.getTokens()) {
                if (token == null || token.isEmpty() || token.trim().isEmpty())
                    continue;
                cnt++;

                if (!set.contains(token))
                    set.add(token);
            }

            lines++;
        }

        log.info("Total number of tokens: [" + cnt + "], lines: [" + lines + "], set size: [" + set.size() + "]");
        log.info("Set:\n" + set);
    }


    @Test
    public void testBuildJointVocabulary1() throws Exception {
        File inputFile = new ClassPathResource("big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);

        VocabCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(iter).tokenizerFactory(t).build();


        /*
            And we pack that transformer into AbstractSequenceIterator
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 0).useAdaGrad(false).setTargetVocabCache(cache).build();

        constructor.buildJointVocabulary(true, false);


        assertEquals(244, cache.numWords());

        assertEquals(0, cache.totalWordOccurrences());
    }


    @Test
    public void testBuildJointVocabulary2() throws Exception {
        File inputFile = new ClassPathResource("big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);

        VocabCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(iter).tokenizerFactory(t).build();


        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 5).useAdaGrad(false).setTargetVocabCache(cache).build();

        constructor.buildJointVocabulary(false, true);

        //        assertFalse(cache.hasToken("including"));

        assertEquals(242, cache.numWords());


        assertEquals("i", cache.wordAtIndex(1));
        assertEquals("it", cache.wordAtIndex(0));

        assertEquals(634303, cache.totalWordOccurrences());
    }

    @Test
    public void testCounter1() throws Exception {
        VocabCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        final List<VocabWord> words = new ArrayList<>();

        words.add(new VocabWord(1, "word"));
        words.add(new VocabWord(2, "test"));
        words.add(new VocabWord(1, "here"));

        Iterable<Sequence<VocabWord>> iterable = new Iterable<Sequence<VocabWord>>() {
            @Override
            public Iterator<Sequence<VocabWord>> iterator() {

                return new Iterator<Sequence<VocabWord>>() {
                    private AtomicBoolean switcher = new AtomicBoolean(true);

                    @Override
                    public boolean hasNext() {
                        return switcher.getAndSet(false);
                    }

                    @Override
                    public Sequence<VocabWord> next() {
                        Sequence<VocabWord> sequence = new Sequence<>(words);
                        return sequence;
                    }

                    @Override
                    public void remove() {
                        throw new UnsupportedOperationException();
                    }
                };
            }
        };


        SequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(iterable).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 0).useAdaGrad(false).setTargetVocabCache(vocabCache).build();

        constructor.buildJointVocabulary(false, true);

        assertEquals(3, vocabCache.numWords());

        assertEquals(1, vocabCache.wordFrequency("test"));
    }

    @Test
    public void testCounter2() throws Exception {
        VocabCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

        final List<VocabWord> words = new ArrayList<>();

        words.add(new VocabWord(1, "word"));
        words.add(new VocabWord(0, "test"));
        words.add(new VocabWord(1, "here"));

        Iterable<Sequence<VocabWord>> iterable = new Iterable<Sequence<VocabWord>>() {
            @Override
            public Iterator<Sequence<VocabWord>> iterator() {

                return new Iterator<Sequence<VocabWord>>() {
                    private AtomicBoolean switcher = new AtomicBoolean(true);

                    @Override
                    public boolean hasNext() {
                        return switcher.getAndSet(false);
                    }

                    @Override
                    public Sequence<VocabWord> next() {
                        Sequence<VocabWord> sequence = new Sequence<>(words);
                        return sequence;
                    }

                    @Override
                    public void remove() {
                        throw new UnsupportedOperationException();
                    }
                };
            }
        };


        SequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(iterable).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 0).useAdaGrad(false).setTargetVocabCache(vocabCache).build();

        constructor.buildJointVocabulary(false, true);

        assertEquals(3, vocabCache.numWords());

        assertEquals(1, vocabCache.wordFrequency("test"));
    }

    /**
     * Here we test basic vocab transfer, done WITHOUT labels
     * @throws Exception
     */
    @Test
    public void testMergedVocab1() throws Exception {
        AbstractCache<VocabWord> cacheSource = new AbstractCache.Builder<VocabWord>().build();

        AbstractCache<VocabWord> cacheTarget = new AbstractCache.Builder<VocabWord>().build();

        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");

        BasicLineIterator underlyingIterator = new BasicLineIterator(resource.getFile());


        SentenceTransformer transformer =
                        new SentenceTransformer.Builder().iterator(underlyingIterator).tokenizerFactory(t).build();

        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> vocabConstructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheSource).build();

        vocabConstructor.buildJointVocabulary(false, true);

        int sourceSize = cacheSource.numWords();
        log.info("Source Vocab size: " + sourceSize);


        VocabConstructor<VocabWord> vocabTransfer = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheTarget).build();

        vocabTransfer.buildMergedVocabulary(cacheSource, false);

        assertEquals(sourceSize, cacheTarget.numWords());
    }

    @Test
    public void testMergedVocabWithLabels1() throws Exception {
        AbstractCache<VocabWord> cacheSource = new AbstractCache.Builder<VocabWord>().build();

        AbstractCache<VocabWord> cacheTarget = new AbstractCache.Builder<VocabWord>().build();

        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");

        BasicLineIterator underlyingIterator = new BasicLineIterator(resource.getFile());


        SentenceTransformer transformer =
                        new SentenceTransformer.Builder().iterator(underlyingIterator).tokenizerFactory(t).build();

        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> vocabConstructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheSource).build();

        vocabConstructor.buildJointVocabulary(false, true);

        int sourceSize = cacheSource.numWords();
        log.info("Source Vocab size: " + sourceSize);

        FileLabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                        .addSourceFolder(new ClassPathResource("/paravec/labeled").getFile()).build();

        transformer = new SentenceTransformer.Builder().iterator(labelAwareIterator).tokenizerFactory(t).build();

        sequenceIterator = new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> vocabTransfer = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(cacheTarget).build();

        vocabTransfer.buildMergedVocabulary(cacheSource, true);

        // those +3 go for 3 additional entries in target VocabCache: labels
        assertEquals(sourceSize + 3, cacheTarget.numWords());

        // now we check index equality for transferred elements
        assertEquals(cacheSource.wordAtIndex(17), cacheTarget.wordAtIndex(17));
        assertEquals(cacheSource.wordAtIndex(45), cacheTarget.wordAtIndex(45));
        assertEquals(cacheSource.wordAtIndex(89), cacheTarget.wordAtIndex(89));

        // we check that newly added labels have indexes beyond the VocabCache index space
        // please note, we need >= since the indexes are zero-based, and sourceSize is not
        assertTrue(cacheTarget.indexOf("Zfinance") > sourceSize - 1);
        assertTrue(cacheTarget.indexOf("Zscience") > sourceSize - 1);
        assertTrue(cacheTarget.indexOf("Zhealth") > sourceSize - 1);
    }
}
