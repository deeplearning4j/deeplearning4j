package org.deeplearning4j.models.word2vec;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.google.common.util.concurrent.AtomicDouble;

import it.unimi.dsi.util.XorShift1024StarRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.actor.*;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.api.Persistable;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.routing.RoundRobinPool;


/**
 * Leveraging a 3 layer neural net with a softmax approach as output,
 * converts a word based on its context and the training examples in to a
 * numeric vector
 * @author Adam Gibson
 *
 */
public class Word2Vec implements Persistable {


    private static final long serialVersionUID = -2367495638286018038L;

    private transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    private transient SentenceIterator sentenceIter;
    private transient DocumentIterator docIter;
    private transient VocabCache cache;

    private int topNSize = 40;
    private int sample = 1;
    //learning rate
    private AtomicDouble alpha = new AtomicDouble(0.025);
    public final static double MIN_ALPHA =  0.001;
    //number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
    private int minWordFrequency = 5;
    //context to use for gathering word frequencies
    private int window = 5;
    private int trainWordsCount = 0;
    //number of neurons per layer
    private int layerSize = 50;
    private transient  RandomGenerator g;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private int size = 0;
    private int words = 0;
    private int allWordsCount = 0;
    private AtomicInteger numSentencesProcessed = new AtomicInteger(0);
    private static ActorSystem trainingSystem;
    private List<String> stopWords;
    private boolean shouldReset = true;
    //number of iterations to run
    private int numIterations = 5;
    public final static String UNK = "UNK";
    private long seed = 123;

    public Word2Vec() {}


    /**
     * Find all words with a similar characters
     * in the vocab
     * @param word the word to compare
     * @param accuracy the accuracy: 0 to 1
     * @return the list of words that are similar in the vocab
     */
    public List<String> similarWordsInVocabTo(String word,double accuracy) {
        List<String> ret = new ArrayList<>();
        for(String s : cache.words()) {
            if(MathUtils.stringSimilarity(word,s) >= accuracy)
                ret.add(s);
        }
        return ret;
    }




    public int indexOf(String word) {
        return cache.indexOf(word);
    }


    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    public float[] getWordVector(String word) {
        int i = this.cache.indexOf(word);
        if(i < 0)
            return cache.vector(UNK).ravel().data();
        return cache.vector(word).ravel().data();
    }

    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    public INDArray getWordVectorMatrix(String word) {
        int i = this.cache.indexOf(word);
        if(i < 0)
            return cache.vector(UNK);
        return cache.vector(word);
    }

    /**
     * Returns the word vector divided by the norm2 of the array
     * @param word the word to get the matrix for
     * @return the looked up matrix
     */
    public INDArray getWordVectorMatrixNormalized(String word) {
        int i = this.cache.indexOf(word);

        if(i < 0)
            return cache.vector(UNK);
        INDArray r =  cache.vector(word);
        return r.div(Nd4j.getBlasWrapper().nrm2(r));
    }


    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    public Collection<String> wordsNearest(String word,int n) {
        INDArray vec = this.getWordVectorMatrix(word);
        if(vec == null)
            return new ArrayList<>();
        Counter<String> distances = new Counter<>();
        for(String s : cache.words()) {
            double sim = similarity(word,s);
            distances.incrementCount(s, sim);
        }


        distances.keepBottomNKeys(n);
        return distances.keySet();

    }

    /**
     * Brings back a list of words that are analagous to the 3 words
     * presented in vector space
     * @param w1
     * @param w2
     * @param w3
     * @return a list of words that are an analogy for the given 3 words
     */
    public List<String> analogyWords(String w1,String w2,String w3) {
        TreeSet<VocabWord> analogies = analogy(w1, w2, w3);
        List<String> ret = new ArrayList<>();
        for(VocabWord w : analogies) {
            String w4 = cache.wordAtIndex(w.getIndex());
            ret.add(w4);
        }
        return ret;
    }




    private void insertTopN(String name, double score, List<VocabWord> wordsEntrys) {
        if (wordsEntrys.size() < topNSize) {
            VocabWord v = new VocabWord(score,name);
            v.setIndex(cache.indexOf(name));
            wordsEntrys.add(v);
            return;
        }
        double min = Float.MAX_VALUE;
        int minOffe = 0;
        int minIndex = -1;
        for (int i = 0; i < topNSize; i++) {
            VocabWord wordEntry = wordsEntrys.get(i);
            if (min > wordEntry.getWordFrequency()) {
                min =  wordEntry.getWordFrequency();
                minOffe = i;
                minIndex = wordEntry.getIndex();
            }
        }

        if (score > min) {
            VocabWord w = new VocabWord(score, VocabWord.PARENT_NODE);
            w.setIndex(minIndex);
            wordsEntrys.set(minOffe,w);
        }

    }

    /**
     * Returns true if the model has this word in the vocab
     * @param word the word to test for
     * @return true if the model has the word in the vocab
     */
    public boolean hasWord(String word) {
        return cache.indexOf(word) >= 0;
    }

    /**
     * Train the model
     */
    public void fit(){
        if(trainingSystem == null)
            trainingSystem = ActorSystem.create();




        buildVocab();
        //save vocab after building
        cache.saveVocab();
        if(stopWords == null)
            readStopWords();


        log.info("Training word2vec multithreaded");

        if(sentenceIter != null)
            sentenceIter.reset();
        if(docIter != null)
            docIter.reset();

        trainingSystem.shutdown();

        final AtomicLong latch = new AtomicLong(0);
        final ActorRef sentenceActor = trainingSystem.actorOf(
                new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(
                        Props.create(SentenceActor.class,this)));


        ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2);

        log.info("Processing sentences...");
        if(getSentenceIter() != null && getSentenceIter().hasNext())
            for(int i = 0; i < numIterations; i++) {
                while(getSentenceIter() != null && getSentenceIter().hasNext()) {

                    final String sentence = sentenceIter.nextSentence();
                    if(sentence == null)
                        continue;


                    service.execute(new Runnable() {

                        /**
                         * When an object implementing interface <code>Runnable</code> is used
                         * to create a thread, starting the thread causes the object's
                         * <code>run</code> method to be called in that separately executing
                         * thread.
                         * <p/>
                         * The general contract of the method <code>run</code> is that it may
                         * take any action whatsoever.
                         *
                         * @see Thread#run()
                         */
                        @Override
                        public void run() {
                            trainSentence(sentence);
                        }
                    });

                    //sentenceActor.tell(new SentenceMessage(sentence,latch),sentenceActor);
                    numSentencesProcessed.incrementAndGet();
                    if(numSentencesProcessed.get() % 100 == 0)
                        log.info("Num sentences processed " + numSentencesProcessed.get());
                }

                if(sentenceIter != null)
                    sentenceIter.reset();
            }


        try {
            service.shutdown();
            service.awaitTermination(1, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        if(docIter != null && docIter.hasNext())
            for(int iter = 0; iter < numIterations; iter++) {
                while (docIter != null && docIter.hasNext()) {
                    InputStream is = docIter.nextDocument();
                    trainSentence(is);
                    numSentencesProcessed.incrementAndGet();
                    if (numSentencesProcessed.get() % 100 == 0)
                        log.info("Num sentences processed " + numSentencesProcessed.get());
                    try {
                        is.close();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                }

                if(docIter != null)
                    docIter.reset();


            }





        while(latch.get() > 0) {
            log.info("Waiting on sentences...Num processed so far " + numSentencesProcessed.get()+ " with latch count at " + latch.get());
            try {
                Thread.sleep(10000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }


        trainingSystem.shutdown();

    }




    /**
     * Train on the given sentence returning a list of vocab words
     * @param is the sentence to fit on
     * @return
     */
    public List<VocabWord> trainSentence(InputStream is) {
        Tokenizer tokenizer = tokenizerFactory.create(is);
        List<VocabWord> sentence2 = new ArrayList<>();

        while(tokenizer.hasMoreTokens()) {
            String next = tokenizer.nextToken();
            if(stopWords.contains(next))
                next = UNK;
            VocabWord word = cache.wordFor(next);
            if(word == null)
                continue;

            sentence2.add(word);

        }


        trainSentence(sentence2);
        return sentence2;
    }


    /**
     * Train on the given sentence returning a list of vocab words
     * @param sentence the sentence to fit on
     * @return
     */
    public List<VocabWord> trainSentence(String sentence) {
        if(sentence.isEmpty())
            return new ArrayList<>();

        Tokenizer tokenizer = tokenizerFactory.create(sentence);
        List<VocabWord> sentence2 = new ArrayList<>();

        while(tokenizer.hasMoreTokens()) {
            String next = tokenizer.nextToken();
            if(stopWords.contains(next))
                next = UNK;
            VocabWord word = cache.wordFor(next);
            if(word == null)
                continue;

            sentence2.add(word);

        }


        trainSentence(sentence2);
        return sentence2;
    }


    /**
     *
     *
     * @param word
     * @return
     */
    public Set<VocabWord> distance(String word) {
        INDArray wordVector = getWordVectorMatrix(word);
        if (wordVector == null) {
            return null;
        }

        INDArray tempVector;
        List<VocabWord> wordEntrys = new ArrayList<>(topNSize);
        for (String name : cache.words()) {
            if (name.equals(word)) {
                continue;
            }

            tempVector = cache.vector(name);
            insertTopN(name, Nd4j.getBlasWrapper().dot(wordVector,tempVector), wordEntrys);
        }
        return new TreeSet<>(wordEntrys);
    }

    /**
     *
     * @return
     */
    public TreeSet<VocabWord> analogy(String word0, String word1, String word2) {
        INDArray wv0 = getWordVectorMatrix(word0);
        INDArray wv1 = getWordVectorMatrix(word1);
        INDArray wv2 = getWordVectorMatrix(word2);


        INDArray wordVector = wv1.sub(wv0).add(wv2);

        if (wv1 == null || wv2 == null || wv0 == null)
            return null;

        INDArray tempVector;
        String name;
        List<VocabWord> wordEntrys = new ArrayList<>(topNSize);
        for (int i = 0; i < cache.numWords(); i++) {
            name = cache.wordAtIndex(i);

            if (name.equals(word0) || name.equals(word1) || name.equals(word2)) {
                continue;
            }


            tempVector = cache.vector(cache.wordAtIndex(i));
            double dist = Nd4j.getBlasWrapper().dot(wordVector,tempVector);
            insertTopN(name, dist, wordEntrys);
        }
        return new TreeSet<>(wordEntrys);
    }


    public void setup() {

        log.info("Building binary tree");
        buildBinaryTree();
        log.info("Resetting weights");
        if(shouldReset)
            resetWeights();
    }


    /**
     * Builds the vocabulary for training
     */
    public void buildVocab() {
        readStopWords();

        if(cache.vocabExists()) {
            log.info("Loading vocab...");
            cache.loadVocab();
            cache.resetWeights();
            return;
        }

        if(trainingSystem == null)
            trainingSystem = ActorSystem.create();





        final AtomicLong semaphore = new AtomicLong(System.currentTimeMillis());
        final AtomicInteger queued = new AtomicInteger(0);

        final ActorRef vocabActor = trainingSystem.actorOf(
                new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(
                        Props.create(VocabActor.class,tokenizerFactory,cache,layerSize,stopWords,semaphore,minWordFrequency)));

		/* all words; including those not in the actual ending index */

        final AtomicInteger latch = new AtomicInteger(0);

        while(docIter != null && docIter.hasNext()) {
            InputStream is = docIter.nextDocument();

            vocabActor.tell(new StreamWork(is,latch),vocabActor);

            queued.incrementAndGet();
            if(queued.get() % 10000 == 0)
                log.info("Sent " + queued);


        }


        while(getSentenceIter() != null && getSentenceIter().hasNext()) {
            String sentence = getSentenceIter().nextSentence();
            if(sentence == null)
                break;
            vocabActor.tell(new VocabWork(latch,sentence), vocabActor);
            queued.incrementAndGet();
            if(queued.get() % 10000 == 0)
                log.info("Sent " + queued);


        }




        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }



        while(latch.get() > 0) {
            log.info("Building vocab...");
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }


        setup();

    }


    /**
     * Create a tsne plot
     */
    public void plotTsne() {
       cache.plotVocab();
    }


    /**
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(final List<VocabWord> sentence) {
        for(int i = 0; i < sentence.size(); i++)
            skipGram(i, sentence);
    }


    /**
     * Train via skip gram
     * @param i
     * @param sentence
     */
    public void skipGram(int i,List<VocabWord> sentence) {
        final VocabWord word = sentence.get(i);
        if(word == null)
            return;
        if(g == null)
            g = new XorShift1024StarRandomGenerator(seed);

        int b = g.nextInt(window);
        int start = Math.max(0, i - window - b);
        int end = i + window + 1 - b;

        for(int a = start; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    VocabWord lastWord = sentence.get(c);
                    iterate(word,lastWord);
                }
            }
        }

    }

    public Map<String,INDArray> toVocabFloat() {
        Map<String,INDArray> ret = new HashMap<>();
        for(int i = 0; i < cache.numWords(); i++) {
            String word = cache.wordAtIndex(i);
            ret.put(word,getWordVectorMatrix(word));
        }

        return ret;

    }





    /**
     * Train the word vector
     * on the given words
     * @param w1 the first word to fit
     */
    public void  iterate(VocabWord w1, VocabWord w2) {
        cache.iterate(w1,w2);
    }




    /* Builds the binary tree for the word relationships */
    private void buildBinaryTree() {
        log.info("Constructing priority queue");
        Huffman huffman = new Huffman(cache.vocabWords());
        huffman.build();

        log.info("Built tree");

    }



    /* reinit weights */
    private void resetWeights() {
        cache.resetWeights();
    }


    /**
     * Returns the similarity of 2 words
     * @param word the first word
     * @param word2 the second word
     * @return a normalized similarity (cosine similarity)
     */
    public double similarity(String word,String word2) {
        if(word.equals(word2))
            return 1.0;

        INDArray vector = getWordVectorMatrix(word);
        INDArray vector2 = getWordVectorMatrix(word2);
        if(vector == null || vector2 == null)
            return -1;
        return  Transforms.cosineSim(vector,vector2);
    }




    @SuppressWarnings("unchecked")
    private void readStopWords() {
        if(this.stopWords != null)
            return;
        this.stopWords = StopWords.getStopWords();


    }




    @Override
    public void write(OutputStream os) {
        try {
            ObjectOutputStream dos = new ObjectOutputStream(os);

            dos.writeObject(this);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public void load(InputStream is) {
        try {
            ObjectInputStream ois = new ObjectInputStream(is);
            Word2Vec vec = (Word2Vec) ois.readObject();
            this.allWordsCount = vec.allWordsCount;
            this.alpha = vec.alpha;
            this.minWordFrequency = vec.minWordFrequency;
            this.numSentencesProcessed = vec.numSentencesProcessed;
            this.sample = vec.sample;
            this.size = vec.size;
            this.stopWords = vec.stopWords;
            this.topNSize = vec.topNSize;
            this.trainWordsCount = vec.trainWordsCount;
            this.window = vec.window;

        }catch(Exception e) {
            throw new RuntimeException(e);
        }



    }




    public int getLayerSize() {
        return layerSize;
    }
    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public int getWindow() {
        return window;
    }



    public int getWords() {
        return words;
    }



    public List<String> getStopWords() {
        return stopWords;
    }

    public  synchronized SentenceIterator getSentenceIter() {
        return sentenceIter;
    }



    public  TokenizerFactory getTokenizerFactory() {
        return tokenizerFactory;
    }




    public  void setTokenizerFactory(TokenizerFactory tokenizerFactory) {
        this.tokenizerFactory = tokenizerFactory;
    }

    public VocabCache getCache() {
        return cache;
    }

    public void setCache(VocabCache cache) {
        this.cache = cache;
    }

    /**
     * Note that calling a setter on this
     * means assumes that this is a training continuation
     * and therefore weights should not be reset.
     * @param sentenceIter
     */
    public void setSentenceIter(SentenceIterator sentenceIter) {
        this.sentenceIter = sentenceIter;
        this.shouldReset = false;
    }



    public static class Builder {
        private int minWordFrequency = 5;
        private int layerSize = 50;
        private SentenceIterator iter;
        private List<String> stopWords = StopWords.getStopWords();
        private int window = 5;
        private TokenizerFactory tokenizerFactory;
        private VocabCache vocabCache;
        private DocumentIterator docIter;
        private float lr = 2.5e-1f;
        private int iterations = 5;
        private long seed = 123;

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }


        public Builder learningRate(float lr) {
            this.lr = lr;
            return this;
        }


        public Builder iterate(DocumentIterator iter) {
            this.docIter = iter;
            return this;
        }

        public Builder vocabCache(VocabCache cache) {
            this.vocabCache = cache;
            return this;
        }

        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder tokenizerFactory(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }



        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public Builder stopWords(List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public Builder windowSize(int window) {
            this.window = window;
            return this;
        }

        public Builder iterate(SentenceIterator iter) {
            this.iter = iter;
            return this;
        }




        public Word2Vec build() {

            if(iter == null) {
                Word2Vec ret = new Word2Vec();
                ret.layerSize = layerSize;
                ret.window = window;
                ret.alpha.set(lr);
                ret.stopWords = stopWords;
                ret.setCache(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null)
                    vocabCache = new InMemoryLookupCache(layerSize);

                ret.docIter = docIter;
                ret.tokenizerFactory = tokenizerFactory;

                return ret;
            }

            else {
                Word2Vec ret = new Word2Vec();
                ret.alpha.set(lr);
                ret.layerSize = layerSize;
                ret.sentenceIter = iter;
                ret.window = window;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setCache(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null)
                    vocabCache = new InMemoryLookupCache(layerSize);

                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }



        }
    }




}