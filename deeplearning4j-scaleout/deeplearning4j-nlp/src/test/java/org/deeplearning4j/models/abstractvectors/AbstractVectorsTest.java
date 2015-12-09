package org.deeplearning4j.models.abstractvectors;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.abstractvectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.abstractvectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
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
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();


        /*
            And we pack that transformer into AbstractSequenceIterator
         */
        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer)
                .build();

        /*
            Now we can build AbstractVectors model, that suits our needs
         */
        AbstractVectors<VocabWord> vectors = new AbstractVectors.Builder<VocabWord>(new VectorsConfiguration())
                .minWordFrequency(3)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(true)
                .build();

        /*
            Now, after all options are set, we just call fit()
         */
        vectors.fit();

        /*
            As soon as fit() exits, model considered built, and we can test it.
            Please note: all similarity context is handled via SequenceElement's labels, so if you're using AbstractVectors to build models for complex
            objects/relations please take care of Labels uniqueness and meaning for yourself.
         */
        double sim = vectors.similarity("day", "night");
        assertTrue(sim > 0.6d);

    }
}