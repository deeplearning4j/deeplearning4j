package org.deeplearning4j.models.sequencevectors;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.canova.api.writable.Writable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.GloVe;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.sequencevectors.graph.enums.NoEdgeHandling;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkDirection;
import org.deeplearning4j.models.sequencevectors.graph.enums.WalkMode;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Graph;
import org.deeplearning4j.models.sequencevectors.graph.primitives.Vertex;
import org.deeplearning4j.models.sequencevectors.iterators.AbstractSequenceIterator;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.transformers.impl.GraphTransformer;
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
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

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

    @Test
    public void testDeepWalk() throws Exception {
        Heartbeat.getInstance().disableHeartbeat();

        AbstractCache<Blogger> vocabCache = new AbstractCache.Builder<Blogger>().build();

        Graph<Blogger, Double> graph =buildGraph();

        GraphTransformer<Blogger> graphTransformer = new GraphTransformer.Builder<Blogger>(graph)
                .setWalkMode(WalkMode.RANDOM)
                .setNoEdgeHandling(NoEdgeHandling.CUTOFF_ON_DISCONNECTED)
                .setWalkLength(40)
                .setWalkDirection(WalkDirection.FORWARD_UNIQUE)
                .shuffleOnReset(true)
                .setVocabCache(vocabCache)
                .build();

        AbstractSequenceIterator<Blogger> sequenceIterator = new AbstractSequenceIterator.Builder<Blogger>(graphTransformer)
                .build();

        WeightLookupTable<Blogger> lookupTable = new InMemoryLookupTable.Builder<Blogger>()
                .lr(0.025)
                .vectorLength(150)
                .useAdaGrad(false)
                .cache(vocabCache)
                .build();


        lookupTable.resetWeights(true);

        SequenceVectors<Blogger> vectors = new SequenceVectors.Builder<Blogger>(new VectorsConfiguration())
                // WeightLookupTable
                .lookupTable(lookupTable)

                // abstract iterator that covers training corpus
                .iterate(sequenceIterator)

                // vocabulary built prior to modelling
                .vocabCache(vocabCache)

                // batchSize is the number of sequences being processed by 1 thread at once
                // this value actually matters if you have iterations > 1
                .batchSize(1000)

                // number of iterations over batch
                .iterations(2)

                // number of iterations over whole training corpus
                .epochs(10)

                // if set to true, vocabulary will be built from scratches internally
                // otherwise externally provided vocab will be used
                .resetModel(false)

                /*
                    These two methods define our training goals. At least one goal should be set to TRUE.
                 */
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(false)

                /*
                    Specifies elements learning algorithms. SkipGram, for example.
                 */
                .elementsLearningAlgorithm(new SkipGram<Blogger>())


                .learningRate(0.025)

                .layerSize(150)

                .sampling(0)

                .negativeSample(0)

                .windowSize(10)

                .workers(6)

                .build();

        vectors.fit();

        vectors.setModelUtils(new FlatModelUtils());

   //     logger.info("12: " + Arrays.toString(vectors.getWordVector("12")));

        double sim = vectors.similarity("12", "72");
        Collection<String> list = vectors.wordsNearest("12", 10);
        logger.info("12->72: " + sim);
        printWords("12", list, vectors);

        assertTrue(sim > 0.10);
        assertFalse(Double.isNaN(sim));
    }


    private List<Blogger> getBloggersFromGraph(Graph<Blogger, Double> graph) {
        List<Blogger> result = new ArrayList<>();

        List<Vertex<Blogger>> bloggers = graph.getVertices(0, graph.numVertices()-1);
        for (Vertex<Blogger> vertex: bloggers) {
            result.add(vertex.getValue());
        }

        return result;
    }

    private static Graph<Blogger, Double> buildGraph() throws IOException, InterruptedException {
        File nodes = new File("/ext/Temp/BlogCatalog/nodes.csv");

        CSVRecordReader reader = new CSVRecordReader(0,",");
        reader.initialize(new FileSplit(nodes));

        List<Blogger> bloggers = new ArrayList<>();
        int cnt = 0;
        while (reader.hasNext()) {
            List<Writable> lines = new ArrayList<>(reader.next());
            Blogger blogger = new Blogger(lines.get(0).toInt());
            blogger.setIndex(cnt);
            bloggers.add(blogger);
            cnt++;
        }

        reader.close();

        Graph<Blogger, Double> graph = new Graph<Blogger, Double>(bloggers, true);

        // load edges
        File edges = new File("/ext/Temp/BlogCatalog/edges.csv");

        reader = new CSVRecordReader(0,",");
        reader.initialize(new FileSplit(edges));

        while (reader.hasNext()) {
            List<Writable> lines = new ArrayList<>(reader.next());
            int from = lines.get(0).toInt();
            int to = lines.get(1).toInt();

            graph.addEdge(from-1, to-1, 1.0, false);
        }
        return graph;
    }

    @Data
    private static class Blogger extends SequenceElement {
        @Getter @Setter private int id;

        public Blogger() {
            super();
        }

        public Blogger(int id) {
            super();
            this.id = id;
        }

        /**
         * This method should return string representation of this SequenceElement, so it can be used for
         *
         * @return
         */
        @Override
        public String getLabel() {
            return String.valueOf(id);
        }

        /**
         * @return
         */
        @Override
        public String toJSON() {
            return null;
        }

        @Override
        public String toString() {
            return super.toString();
        }
    }

    private static void printWords(String target, Collection<String> list, SequenceVectors vec) {
        System.out.println("Words close to ["+target+"]: ");
        for (String word: list) {
            double sim = vec.similarity(target, word);
            System.out.print("'"+ word+"': ["+ sim+"], ");
        }
        System.out.print("\n");
    }
}