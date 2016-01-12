package org.deeplearning4j.models.word2vec.wordstore;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
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

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class VocabConstructorTest {

    protected static final Logger log = LoggerFactory.getLogger(VocabConstructorTest.class);

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testVocab() throws Exception {
        File inputFile = new ClassPathResource("big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Set<String> set = new HashSet<>();
        int lines = 0;
        int cnt = 0;
        while (iter.hasNext()) {
            Tokenizer tok = t.create(iter.nextSentence());
            for (String token: tok.getTokens()) {
                if (token == null || token.isEmpty() || token.trim().isEmpty()) continue;
                    cnt++;

                if (!set.contains(token))
                    set.add(token);
            }

            lines++;
        }

        log.info("Total number of tokens: [" + cnt + "], lines: [" + lines+"], set size: ["+ set.size() +"]");
        log.info("Set:\n" + set);
    }


    @Test
    public void testBuildJointVocabulary1() throws Exception {
        File inputFile = new ClassPathResource("big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(iter)
                .tokenizerFactory(t)
                .build();


        /*
            And we pack that transformer into AbstractSequenceIterator
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer)
                .build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 0)
                .useAdaGrad(false)
                .setTargetVocabCache(cache)
                .build();

        constructor.buildJointVocabulary(true, false);


        assertEquals(244, cache.numWords());

        assertEquals(0, cache.totalWordOccurrences());
    }


    @Test
    public void testBuildJointVocabulary2() throws Exception {
        File inputFile = new ClassPathResource("big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(iter)
                .tokenizerFactory(t)
                .build();


        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer)
                .build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 5)
                .useAdaGrad(false)
                .setTargetVocabCache(cache)
                .build();

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
                };
            };
        };


        SequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(iterable).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 0)
                .useAdaGrad(false)
                .setTargetVocabCache(vocabCache)
                .build();

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
                };
            };
        };


        SequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<>(iterable).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 0)
                .useAdaGrad(false)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);

        assertEquals(3, vocabCache.numWords());

        assertEquals(1, vocabCache.wordFrequency("test"));
    }
/*
    @Test
    public void testVocabTransfer1() throws Exception {

        InMemoryLookupCache cache = new InMemoryLookupCache();

        VocabularyHolder holder = new VocabularyHolder.Builder()
                .externalCache(cache)
                .build();

        holder.addWord("testerz");

        holder.transferBackToVocabCache();

        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(iter, 5)
                .useAdaGrad(false)
                .setTargetVocabCache(cache)
                .build();

        constructor.buildJointVocabulary(false, true);

        assertEquals(634061, cache.totalWordOccurrences());
        assertEquals(243, cache.numWords());

        assertEquals("it", cache.wordAtIndex(0));
        assertEquals("i", cache.wordAtIndex(1));

        assertNotEquals(null, cache.wordFor("testerz"));
        assertEquals(1, cache.wordFrequency("testerz"));
    }

    @Test
    public void testVocabBuildingWithLabels1() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource generator = new LabelsSource("SNTX_");

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(new SentenceIteratorConverter(iter, generator), 5)
                .useAdaGrad(false)
                .fetchLabels(true)
                .build();

        VocabCache cache = constructor.buildJointVocabulary(false, true);


        log.info("Total words in vocab: ["+ cache.numWords() + "], Total word occurencies: [" + cache.totalWordOccurrences() + "]");

        assertEquals(97168, generator.getLabels().size());
        assertEquals(97410, cache.numWords());
        assertEquals(634061, cache.totalWordOccurrences());

        assertTrue(cache.containsWord("SNTX_8"));
        assertEquals(1, cache.wordFrequency("SNTX_8"));

    }

    @Test
    public void testVocabBuildingWithLabels2() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache();

        LabelsSource generator = new LabelsSource("SNTX_");

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(new SentenceIteratorConverter(iter, generator), 5)
                .setTargetVocabCache(cache)
                .useAdaGrad(false)
                .fetchLabels(true)
                .build();

        constructor.buildJointVocabulary(false, true);


        log.info("Total words in vocab: ["+ cache.numWords() + "]");


        assertEquals(97168, generator.getLabels().size());
        assertEquals(97410, cache.numWords());
        assertEquals(634061, cache.totalWordOccurrences());

        assertTrue(cache.containsWord("SNTX_8"));
        assertEquals(1, cache.wordFrequency("SNTX_8"));

    }
    */
}