package org.deeplearning4j.models.abstractvectors;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.abstractvectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.abstractvectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.junit.Before;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.*;

/**
 *
 * @author raver119@gmail.com
 */
public class AbstractVectorsTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testAbstractW2VModel() throws Exception {
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();


        /*
            First we build line iterator
         */
        BasicLineIterator underlyingIterator = new BasicLineIterator(file);


        /*
            Now we need the way to convert lines into Sequences of VocabWords.
            In this example that's SentenceTransformer
         */
        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .build();

        /*
            And we pack that transformer into AbstractSequenceIterator
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer)
                .build();

        AbstractVectors<VocabWord> vectors = new AbstractVectors.Builder<VocabWord>(new VectorsConfiguration())
                .iterate(sequenceIterator)
                .build();

        /*
            Now, after all options are set, we just call fit()
         */
        vectors.fit();

        double sim = vectors.similarity("day", "night");
        assertTrue(sim > 0.6d);

    }
}