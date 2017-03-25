package org.deeplearning4j.models.word2vec;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;

import static org.junit.Assert.*;

/**
 *
 * Basic w2v model based on example raw text.
 *
 *
 * Created by raver119@gmail.com on 08.11.15.
 */
public class StandaloneWord2VecTest {

    private static Logger log = LoggerFactory.getLogger(StandaloneWord2VecTest.class);

    @Test
    public void testFit() throws Exception {
        String filePath = "src/test/resources/w2v/raw_sentences.txt";
        SentenceIterator iter = UimaSentenceIterator.createWithPath(filePath);

        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache(true);

        WeightLookupTable table = new InMemoryLookupTable.Builder()
                .vectorLength(200)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        log.info("Building model....");
        StandaloneWord2Vec vec = new StandaloneWord2Vec.Builder()
                .minWordFrequency(3).iterations(2)
                .layerSize(200).lookupTable(table)
                .stopList(new ArrayList<String>())
                .vocabCache(cache).seed(42)
                .numThreads(5)
                .windowSize(5).iterator(iter).tokenizerFactory(t).build();

        log.info("Fitting StandaloneWord2Vec model....");
        vec.fit();

        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info("Nearest words to the word 'day': " + lst);

        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("year"));

        assertFalse(lst.contains(null));
   }
}