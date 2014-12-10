package org.deeplearning4j.models.glove;

import static org.junit.Assert.*;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.StringCleaning;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class CoOccurrencesTest {
    private static Logger log = LoggerFactory.getLogger(CoOccurrencesTest.class);
    private CoOccurrences coOccurrences;
    private VocabCache vocabCache;
    private SentenceIterator iter;
    private TextVectorizer textVectorizer;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

    @Before
    public void before() throws Exception {
        ClassPathResource resource = new ClassPathResource("other/oneline.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return StringCleaning.stripPunct(sentence);
            }
        });


    }


    @Test
    public void testCoOccurrences() {
        if(vocabCache == null)
            vocabCache = new InMemoryLookupCache();

        if(textVectorizer == null) {
            textVectorizer = new TfidfVectorizer.Builder().tokenize(tokenizerFactory)
                    .cache(vocabCache).iterate(iter).minWords(1).stopWords(new ArrayList<String>())
                    .build();

            textVectorizer.fit();
        }

        if(coOccurrences == null) {
            coOccurrences = new CoOccurrences.Builder()
                    .cache(vocabCache).iterate(iter).symmetric(false)
                    .tokenizer(tokenizerFactory).windowSize(15)
                    .build();

            coOccurrences.fit();
            assertEquals(16,coOccurrences.getCoOCurreneCounts().totalSize());
            log.info(coOccurrences.getCoOCurreneCounts().toString());

        }

    }


    @Test
    public void testWeights() throws Exception {
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });


        vocabCache = new InMemoryLookupCache();

        textVectorizer = new TfidfVectorizer.Builder().tokenize(tokenizerFactory)
                .cache(vocabCache).iterate(iter).minWords(1).stopWords(new ArrayList<String>())
                .build();

        textVectorizer.fit();

        coOccurrences = new CoOccurrences.Builder()
                .cache(vocabCache).iterate(iter).symmetric(false)
                .tokenizer(tokenizerFactory).windowSize(15)
                .build();

        coOccurrences.fit();
        List<String> occurrences = IOUtils.readLines(new ClassPathResource("big/coc.txt").getInputStream());
        for(int i = 0; i < occurrences.size(); i++) {
            String[] split = occurrences.get(i).split(" ");
            //punctuation can vary: not a huge deal here
            if (split.length < 3 || StringCleaning.stripPunct(split[0]).isEmpty() || StringCleaning.stripPunct(split[1]).isEmpty())
                continue;
            double count = coOccurrences.count(split[0], split[1]);
            if(count == 0)
                count = coOccurrences.count(split[1], split[0]);
            //weighting doesn't need to be exact but should be reasonably close
            assertEquals("Failed on " + split[0] + " " + split[1], count, Double.parseDouble(split[2]), 5);
        }






    }


    @Test
    public void testNumOccurrences() throws Exception {
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });


        vocabCache = new InMemoryLookupCache();

        textVectorizer = new TfidfVectorizer.Builder().tokenize(tokenizerFactory)
                .cache(vocabCache).iterate(iter).minWords(1).stopWords(new ArrayList<String>())
                .build();

        textVectorizer.fit();

        coOccurrences = new CoOccurrences.Builder()
                .cache(vocabCache).iterate(iter).symmetric(false)
                .tokenizer(tokenizerFactory).windowSize(15)
                .build();

        coOccurrences.fit();
        List<String> occurrences = IOUtils.readLines(new ClassPathResource("big/occurrences.txt").getInputStream());
        for(int i = 0; i < occurrences.size(); i++) {
            String[] split = occurrences.get(i).split(" ");
            assertEquals(i,Integer.parseInt(split[0]));
            assertEquals("Failed on id " + i,Double.parseDouble(split[1]),coOccurrences.getSentenceOccurrences().getCount(i),6);
        }

        log.info(String.valueOf(coOccurrences.numCoOccurrences()));





    }

    @Test
    public void testTokens() throws Exception {
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });
        List<String> lines = IOUtils.readLines(new ClassPathResource(("big/tokens.txt")).getInputStream());
        int count = 0;
        while(iter.hasNext()) {
            List<String> tokens = tokenizerFactory.create(iter.nextSentence()).getTokens();
            String[] split = lines.get(count).split(" ");
            int count2 = Integer.parseInt(split[0]);
            assertEquals(count,count2);
            assertEquals("Sentence " + count,Integer.parseInt(split[1]),tokens.size());
            count++;
        }


    }
}
