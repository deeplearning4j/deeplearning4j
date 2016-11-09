package org.deeplearning4j.models.sequencevectors.transformers.impl.iterables;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class ParallelTransformerIteratorTest {
    private TokenizerFactory factory = new DefaultTokenizerFactory();

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void hasNext() throws Exception {
        SentenceIterator iterator = new BasicLineIterator(new ClassPathResource("/big/raw_sentences.txt").getFile());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(iterator)
                .allowMultithreading(true)
                .tokenizerFactory(factory)
                .build();

        Iterator<Sequence<VocabWord>> iter = transformer.iterator();
        int cnt = 0;
        while (iter.hasNext()) {
            Sequence<VocabWord> sequence = iter.next();
            assertNotEquals("Failed on [" + cnt + "] iteration", null, sequence);
            assertNotEquals("Failed on [" + cnt + "] iteration", 0, sequence.size());
            cnt++;
        }

        assertEquals(97162, cnt);
    }

}