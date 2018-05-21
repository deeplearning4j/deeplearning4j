package org.deeplearning4j.models.glove;

import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author raver119@gmail.com
 */
public class AbstractCoOccurrencesTest {

    private static final Logger log = LoggerFactory.getLogger(AbstractCoOccurrencesTest.class);

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testFit1() throws Exception {
        ClassPathResource resource = new ClassPathResource("other/oneline.txt");
        File file = resource.getFile();

        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();
        BasicLineIterator underlyingIterator = new BasicLineIterator(file);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer =
                        new SentenceTransformer.Builder().iterator(underlyingIterator).tokenizerFactory(t).build();

        AbstractSequenceIterator<VocabWord> sequenceIterator =
                        new AbstractSequenceIterator.Builder<>(transformer).build();

        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                        .addSource(sequenceIterator, 1).setTargetVocabCache(vocabCache).build();

        constructor.buildJointVocabulary(false, true);

        AbstractCoOccurrences<VocabWord> coOccurrences = new AbstractCoOccurrences.Builder<VocabWord>()
                        .iterate(sequenceIterator).vocabCache(vocabCache).symmetric(false).windowSize(15).build();

        coOccurrences.fit();

        //List<Pair<VocabWord, VocabWord>> list = coOccurrences.i();
        Iterator<Pair<Pair<VocabWord, VocabWord>, Double>> iterator = coOccurrences.iterator();
        assertNotEquals(null, iterator);
        int cnt = 0;

        List<Pair<VocabWord, VocabWord>> list = new ArrayList<>();
        while (iterator.hasNext()) {
            Pair<Pair<VocabWord, VocabWord>, Double> pair = iterator.next();
            list.add(pair.getFirst());
            cnt++;
        }


        log.info("CoOccurrences: " + list);

        assertEquals(16, list.size());
        assertEquals(16, cnt);
    }
}
