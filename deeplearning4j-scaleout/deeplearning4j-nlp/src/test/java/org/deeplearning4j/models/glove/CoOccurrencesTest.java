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

package org.deeplearning4j.models.glove;

import static org.junit.Assert.*;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.LegacyTfidfVectorizer;
import org.deeplearning4j.berkeley.Pair;
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

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class CoOccurrencesTest {
    private static final Logger log = LoggerFactory.getLogger(CoOccurrencesTest.class);
    private CoOccurrences coOccurrences;
    private VocabCache vocabCache;
    private SentenceIterator iter;
    private TextVectorizer textVectorizer;
    private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

    @Before
    public void before() throws Exception {
        FileUtils.deleteDirectory(new File("word2vec-index"));
        ClassPathResource resource = new ClassPathResource("other/oneline.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return StringCleaning.stripPunct(sentence);
            }
        });
        vocabCache = new InMemoryLookupCache();

        textVectorizer = new LegacyTfidfVectorizer.Builder().tokenize(tokenizerFactory)
                .cache(vocabCache).iterate(iter).minWords(1).stopWords(new ArrayList<String>())
                .build();

        textVectorizer.fit();

        coOccurrences = new CoOccurrences.Builder()
                .cache(vocabCache).iterate(iter).symmetric(false)
                .tokenizer(tokenizerFactory).windowSize(15)
                .build();

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

    @Test
    public void testOriginalCoOccurrences() throws Exception {
        coOccurrences.fit();

        List<Pair<String, String>> list = coOccurrences.coOccurrenceList();
        log.info("Number of CoOccurrences: ["+ list.size()+"]");
        log.info("CoOccurences: " + list);
    }
}
