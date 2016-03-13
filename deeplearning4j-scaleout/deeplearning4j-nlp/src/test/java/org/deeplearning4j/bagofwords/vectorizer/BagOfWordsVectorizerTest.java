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


import static org.junit.Assume.*;

import org.apache.commons.io.FileUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 *@author Adam Gibson
 */
public class BagOfWordsVectorizerTest {

    private static final Logger log = LoggerFactory.getLogger(BagOfWordsVectorizerTest.class);
    private InvertedIndex index;
    private VocabCache cache;

    @Before
    public void before() {
        cache = new InMemoryLookupCache();
        index = new LuceneInvertedIndex.Builder().indexDir(new File("bagofwords"))
                .cache(cache).batchSize(5)
                .cacheInRam(false).build();

    }

    @After
    public void after() throws Exception {
        FileUtils.deleteDirectory(new File("bagofwords"));
    }


    @Test
    public void testBagOfWordsVectorizer() throws Exception {
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> labels = Arrays.asList("label1", "label2");
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        TextVectorizer vectorizer = new LegacyBagOfWordsVectorizer.Builder().index(index)
                .cache(cache)
                .minWords(1).stopWords(new ArrayList<String>()).cleanup(true)
                .tokenize(tokenizerFactory).iterate(iter).labels(labels).build();
        vectorizer.fit();
        VocabWord word = (VocabWord) vectorizer.vocab().wordFor("file.");
        assumeNotNull(word);
        assertEquals(word,vectorizer.vocab().tokenFor("file."));
        assertEquals(2,vectorizer.index().numDocuments());


    }


}
