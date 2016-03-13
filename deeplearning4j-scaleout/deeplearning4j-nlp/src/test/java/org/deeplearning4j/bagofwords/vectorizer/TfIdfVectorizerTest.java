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

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeNotNull;

/**
 * @author Adam Gibson
 */
public class TfIdfVectorizerTest {

    private static final Logger log = LoggerFactory.getLogger(TfIdfVectorizerTest.class);

    private InvertedIndex<VocabWord> index;
    private  VocabCache<VocabWord> cache;

    @Before
    public void before() throws Exception {
        FileUtils.deleteDirectory(new File("tfidf"));

        cache = new InMemoryLookupCache();
        index = new LuceneInvertedIndex.Builder<VocabWord>().cache(cache)
                .indexDir(new File("tfidf"))
                .batchSize(5)
                .cacheInRam(false)
                .build();

    }

    @After
    public void after() throws Exception {
        if(index != null)
            index.cleanup();

        FileUtils.deleteDirectory(new File("tfidf"));
    }

    @Test
    public void testTfIdfVectorizer() throws Exception {
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> docStrings = new ArrayList<>();

        while(iter.hasNext())
            docStrings.add(iter.nextSentence());

        iter.reset();

        List<String> labels = Arrays.asList("label1","label2");
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizer = new LegacyTfidfVectorizer.Builder()
                .minWords(1).index(index).cache(cache)
                .stopWords(new ArrayList<String>())
                .tokenize(tokenizerFactory).labels(labels)
                .iterate(iter).build();

        vectorizer.fit();
        try {
            vectorizer.vectorize("",null);
            fail("Vectorizer should receive non-null label.");
        } catch (IllegalArgumentException e) {
            ;
        }

        VocabWord word = (VocabWord) vectorizer.vocab().wordFor("file");
        assumeNotNull(word);
        assertEquals(word,vectorizer.vocab().tokenFor("file"));


        int[] docs = vectorizer.index().allDocs();
        InvertedIndex<VocabWord> localIndex  =  vectorizer.index();
        for(int i : docs) {
            StringBuilder sb = new StringBuilder();
            List<VocabWord> doc = localIndex.document(i);
            for(VocabWord w : doc)
                sb.append(" " + w.getWord());
            log.info("Doc " + sb.toString());
        }

        assertEquals(docStrings.size(),docs.length);
        assertEquals(docStrings.size(), localIndex.documents(word).length);

    }
}
