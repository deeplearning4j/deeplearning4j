package org.deeplearning4j.models.word2vec.wordstore;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 22.11.15.
 */
public class VocabConstructorTest {

    protected static final Logger log = LoggerFactory.getLogger(VocabConstructorTest.class);

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testBuildJointVocabulary1() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabConstructor constructor = new VocabConstructor.Builder()
                .setTokenizerFactory(t)
                .addSource(iter, 0)
                .useAdaGrad(false)
                .build();

        VocabCache cache = constructor.buildJointVocabulary(true, false);

        assertEquals(0, cache.totalWordOccurrences());
        assertEquals(244, cache.numWords());
    }

    @Test
    public void testBuildJointVocabulary2() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(iter, 5)
                .useAdaGrad(false)
                .build();

        VocabCache cache = constructor.buildJointVocabulary(false, true);

        assertEquals(634061, cache.totalWordOccurrences());
        assertEquals(242, cache.numWords());

        assertEquals("it", cache.wordAtIndex(0));
        assertEquals("i", cache.wordAtIndex(1));
    }

    @Test
    public void testVocabTransfer1() throws Exception {

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

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
}