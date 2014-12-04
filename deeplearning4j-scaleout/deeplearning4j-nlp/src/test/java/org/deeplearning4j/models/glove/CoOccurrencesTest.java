package org.deeplearning4j.models.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;

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
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());


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
                    .cache(vocabCache).iterate(iter)
                    .tokenizer(tokenizerFactory).windowSize(15)
                    .build();

            coOccurrences.fit();

            log.info(coOccurrences.getCoOCurreneCounts().toString());

        }

    }


}
