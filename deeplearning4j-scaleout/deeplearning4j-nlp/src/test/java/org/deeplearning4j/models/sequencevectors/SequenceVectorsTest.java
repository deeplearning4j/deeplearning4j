package org.deeplearning4j.models.sequencevectors;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.learning.impl.elements.GloVe;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabConstructor;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

import static org.junit.Assert.*;

/**
 *
 * @author raver119@gmail.com
 */
public class SequenceVectorsTest {

    protected static final Logger logger = LoggerFactory.getLogger(SequenceVectorsTest.class);

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
            Now we should build vocabulary out of sequence iterator.
            We can skip this phase, and just set SequenceVectors.resetModel(TRUE), and vocabulary will be mastered internally
        */
        VocabConstructor<VocabWord> constructor = new VocabConstructor.Builder<VocabWord>()
                .addSource(sequenceIterator, 5)
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

         /*
             reset model is viable only if you're setting SequenceVectors.resetModel() to false
             if set to True - it will be called internally
        */
        lookupTable.resetWeights(true);

        /*
            Now we can build SequenceVectors model, that suits our needs
         */
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                // minimum number of occurencies for each element in training corpus. All elements below this value will be ignored
                // Please note: this value has effect only if resetModel() set to TRUE, for internal model building. Otherwise it'll be ignored, and actual vocabulary content will be used
                .minWordFrequency(5)

                // WeightLookupTable
                .lookupTable(lookupTable)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // vocabulary built prior to modelling
                .vocabCache(vocabCache)

                // batchSize is the number of sequences being processed by 1 thread at once
                // this value actually matters if you have iterations > 1
                .batchSize(250)

                // number of iterations over batch
                .iterations(1)

                // number of iterations over whole training corpus
                .epochs(1)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(false)


                /*
                    These two methods define our training goals. At least one goal should be set to TRUE.
                 */
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)

                .build();

        /*
            Now, after all options are set, we just call fit()
         */
        vectors.fit();

        /*
            As soon as fit() exits, model considered built, and we can test it.
            Please note: all similarity context is handled via SequenceElement's labels, so if you're using SequenceVectors to build models for complex
            objects/relations please take care of Labels uniqueness and meaning for yourself.
         */
        double sim = vectors.similarity("day", "night");
        logger.info("Day/night similarity: " + sim);
        assertTrue(sim > 0.6d);

        Collection<String> labels = vectors.wordsNearest("day", 10);
        logger.info("Nearest labels to 'day': " + labels);
    }

    @Test
    public void testInternalVocabConstruction() throws Exception {
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();

        BasicLineIterator underlyingIterator = new BasicLineIterator(file);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();

        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer)
                .build();

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .minWordFrequency(5)
                .iterate(sequenceIterator)
                .batchSize(250)
                .iterations(1)
                .epochs(1)
                .resetModel(false)
                .trainElementsRepresentation(true)
                .build();

        vectors.fit();

        double sim = vectors.similarity("day", "night");
        logger.info("Day/night similarity: " + sim);
        assertTrue(sim > 0.6d);

        Collection<String> labels = vectors.wordsNearest("day", 10);
        logger.info("Nearest labels to 'day': " + labels);
    }

    @Test
    public void testElementsLearningAlgo1() throws Exception {
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .minWordFrequency(5)
                .batchSize(250)
                .iterations(1)
                .elementsLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram")
                .epochs(1)
                .resetModel(false)
                .trainElementsRepresentation(true)
                .build();
    }

    @Test
    public void testSequenceLearningAlgo1() throws Exception {
        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration())
                .minWordFrequency(5)
                .batchSize(250)
                .iterations(1)
                .sequenceLearningAlgorithm("org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW")
                .epochs(1)
                .resetModel(false)
                .trainElementsRepresentation(false)
                .build();
    }

    @Test
    public void testGlove1() throws Exception {
        logger.info("Max available memory: " + Runtime.getRuntime().maxMemory());
        ClassPathResource resource = new ClassPathResource("big/raw_sentences.txt");
        File file = resource.getFile();

        BasicLineIterator underlyingIterator = new BasicLineIterator(file);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        SentenceTransformer transformer = new SentenceTransformer.Builder()
                .iterator(underlyingIterator)
                .tokenizerFactory(t)
                .build();

        AbstractSequenceIterator<VocabWord> sequenceIterator = new AbstractSequenceIterator.Builder<VocabWord>(transformer)
                .build();

        VectorsConfiguration configuration = new VectorsConfiguration();
        configuration.setWindow(5);
        configuration.setLearningRate(0.06);
        configuration.setLayersSize(100);


        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(configuration)
                .iterate(sequenceIterator)
                .iterations(1)
                .epochs(45)
                .elementsLearningAlgorithm(new GloVe.Builder<VocabWord>()
                        .shuffle(true)
                        .symmetric(true)
                        .learningRate(0.05)
                        .alpha(0.75)
                        .xMax(100.0)
                        .build())
                .resetModel(true)
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)
                .build();

        vectors.fit();

        double sim = vectors.similarity("day", "night");
        logger.info("Day/night similarity: " + sim);


        sim = vectors.similarity("day", "another");
        logger.info("Day/another similarity: " + sim);

        sim = vectors.similarity("night", "year");
        logger.info("Night/year similarity: " + sim);

        sim = vectors.similarity("night", "me");
        logger.info("Night/me similarity: " + sim);

        sim = vectors.similarity("day", "know");
        logger.info("Day/know similarity: " + sim);

        sim = vectors.similarity("best", "police");
        logger.info("Best/police similarity: " + sim);

        Collection<String> labels = vectors.wordsNearest("day", 10);
        logger.info("Nearest labels to 'day': " + labels);


        sim = vectors.similarity("day", "night");
        assertTrue(sim > 0.6d);
    }
}