package org.deeplearning4j.models.abstractvectors;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.abstractvectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.abstractvectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

import static org.junit.Assert.*;

/**
 *
 * @author raver119@gmail.com
 */
public class AbstractVectorsTest {

    protected static final Logger logger = LoggerFactory.getLogger(AbstractVectorsTest.class);

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testAbstractW2VModel() throws Exception {
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();

        AbstractCache<VocabWord> vocabCache = new AbstractCache.Builder<VocabWord>().build();

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
            Now we should build vocabulary out of sequence iterator
        */
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 5)
                .useAdaGrad(false)
                .setTargetVocabCache(vocabCache)
                .build();

        constructor.buildJointVocabulary(false, true);

        assertEquals(242, vocabCache.numWords());

        assertEquals(634303, vocabCache.totalWordOccurrences());


        /*
            Time to build WeightLookupTable instance for our new model
        */

        WeightLookupTable<VocabWord> lookupTable = new InMemoryLookupTable.Builder<VocabWord>()
                .lr(0.025)
                .vectorLength(150)
                .useAdaGrad(false)
                .cache(vocabCache)
                .build();

        lookupTable.resetWeights(true);

        /*
            Now we can build AbstractVectors model, that suits our needs
         */
        AbstractVectors<VocabWord> vectors = new AbstractVectors.Builder<VocabWord>(new VectorsConfiguration())
                .minWordFrequency(3)

                // WeightLookupTable
                .setLookupTable(lookupTable)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // vocabulary built prior to modelling
                .setVocabCache(vocabCache)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(false)


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
        logger.info("Day/night similarity: " + sim);
        assertTrue(sim > 0.6d);

    }
}