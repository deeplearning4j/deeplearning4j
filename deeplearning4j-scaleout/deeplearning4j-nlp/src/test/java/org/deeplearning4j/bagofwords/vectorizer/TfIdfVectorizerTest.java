package org.deeplearning4j.bagofwords.vectorizer;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeNotNull;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.junit.Before;
import org.junit.After;

import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.*;

/**
 * @author Adam Gibson
 */
public class TfIdfVectorizerTest {

    private static Logger log = LoggerFactory.getLogger(TfIdfVectorizerTest.class);

    private InvertedIndex index;
    private  VocabCache cache;

    @Before
    public void before() throws Exception {
        FileUtils.deleteDirectory(new File("tfidf"));

        cache = new InMemoryLookupCache();
        index = new LuceneInvertedIndex.Builder().cache(cache)
                .indexDir(new File("tfidf"))
                .batchSize(5)
                .cacheInRam(false).build();

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

        TextVectorizer vectorizer = new TfidfVectorizer.Builder()
                .minWords(1).index(index).cache(cache)
                .stopWords(new ArrayList<String>())
                .tokenize(tokenizerFactory).labels(labels)
                .iterate(iter).build();

        vectorizer.fit();

        VocabWord word = vectorizer.vocab().wordFor("file");
        assumeNotNull(word);
        assertEquals(word,vectorizer.vocab().tokenFor("file"));


        int[] docs = vectorizer.index().allDocs();
        for(int i : docs) {
            StringBuffer sb = new StringBuffer();
            List<VocabWord> doc = vectorizer.index().document(i);
            for(VocabWord w : doc)
                sb.append(" " + w.getWord());
            log.info("Doc " + sb.toString());
        }

        assertEquals(docStrings.size(),docs.length);
        assertEquals(docStrings.size(), vectorizer.index().documents(word).length);



    }


}
