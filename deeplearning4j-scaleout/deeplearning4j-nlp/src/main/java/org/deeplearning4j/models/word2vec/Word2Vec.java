package org.deeplearning4j.models.word2vec;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Counter;
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
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



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
    private int batchSize = 1000;
    private int topNSize = 40;
    private double sample = 0;
    private long totalWords = 1;
    private AtomicInteger rateOfChange = new AtomicInteger(0);
    //learning rate
    private AtomicDouble alpha = new AtomicDouble(0.025);
    //number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
    private int minWordFrequency = 5;
    //context to use for gathering word frequencies
    private int window = 5;
    //number of neurons per layer
    private int layerSize = 50;
    private transient  RandomGenerator g;
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private List<String> stopWords;
    private boolean shouldReset = true;
    //number of iterations to run
    private int numIterations = 1;
    public final static String UNK = "UNK";
    private long seed = 123;
    private boolean saveVocab = false;
    private double minLearningRate = 0.01;
    private TextVectorizer vectorizer;
    private int learningRateDecayWords = 10000;
    private boolean useAdaGrad = false;


    public Word2Vec() {}






    /**
     * Accuracy based on questions which are a space separated list of strings
     * where the first word is the query word, the next 2 words are negative,
     * and the last word is the predicted word to be nearest
     * @param questions the questions to ask
     * @return the accuracy based on these questions
     */
    public  Map<String,Double> accuracy(List<String> questions) {
        Map<String,Double> accuracy = new HashMap<>();
        Counter<String> right = new Counter<>();
        for(String s : questions) {
            if(s.startsWith(":")) {
                double correct = right.getCount("correct");
                double wrong = right.getCount("wrong");
                double accuracyRet = 100.0 * correct / (correct / wrong);
                accuracy.put(s,accuracyRet);
                right.clear();
            }
            else {
                String[] split = s.split(" ");
                String word = split[0];
                List<String> positive = Arrays.asList(word);
                List<String> negative = Arrays.asList(split[1],split[2]);
                String predicted = split[3];
                String w = wordsNearest(positive,negative,1).iterator().next();
                if(predicted.equals(w))
                    right.incrementCount("right",1.0);
                else
                    right.incrementCount("wrong",1.0);

            }
        }

        return accuracy;
    }



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
    public double[] getWordVector(String word) {
        int i = this.cache.indexOf(word);
        if(i < 0)
            return cache.vector(UNK).ravel().data().asDouble();
        return cache.vector(word).ravel().data().asDouble();
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
     * Words nearest based on positive and negative words
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    public Collection<String> wordsNearest(List<String> positive,List<String> negative,int top) {
        INDArray words = Nd4j.create(positive.size() + negative.size(),layerSize);
        int row = 0;
        for(String s : positive) {
            words.putRow(row++,cache.vector(s));
        }

        for(String s : negative) {
            words.putRow(row++,cache.vector(s).mul(-1));
        }

        INDArray mean = words.mean(0);
        if(cache instanceof  InMemoryLookupCache) {
            InMemoryLookupCache l = (InMemoryLookupCache) cache;
            INDArray syn0 = l.getSyn0();
            INDArray weights = syn0.norm2(0).rdivi(1).muli(mean);
            INDArray distances = syn0.mulRowVector(weights).sum(1);
            INDArray[] sorted = Nd4j.sortWithIndices(distances,0,false);
            INDArray sort = sorted[0];
            List<String> ret = new ArrayList<>();
            if(top > sort.length())
                top = sort.length();
            //there will be a redundant word
            for(int i = 0; i < top + 1; i++) {
                ret.add(cache.wordAtIndex(sort.getInt(i)));
            }


            return ret;
        }

        Counter<String> distances = new Counter<>();

        for(String s : cache.words()) {
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(mean,otherVec);
            distances.incrementCount(s, sim);
        }


        distances.keepTopNKeys(top);
        return distances.keySet();


    }


    /**
     * Get the top n words most similar to the given word
     * @param word the word to compare
     * @param n the n to get
     * @return the top n words
     */
    public Collection<String> wordsNearest(String word,int n) {
        INDArray vec = Transforms.unitVec(this.getWordVectorMatrix(word));


        if(cache instanceof  InMemoryLookupCache) {
            InMemoryLookupCache l = (InMemoryLookupCache) cache;
            INDArray syn0 = l.getSyn0();
            INDArray weights = syn0.norm2(0).rdivi(1).muli(vec);
            INDArray distances = syn0.mulRowVector(weights).sum(1);
            INDArray[] sorted = Nd4j.sortWithIndices(distances,0,false);
            INDArray sort = sorted[0];
            List<String> ret = new ArrayList<>();
            VocabWord word2 = cache.wordFor(word);
            if(n > sort.length())
                n = sort.length();
            //there will be a redundant word
            for(int i = 0; i < n + 1; i++) {
                if(sort.getInt(i) == word2.getIndex())
                    continue;
                ret.add(cache.wordAtIndex(sort.getInt(i)));
            }


            return ret;
        }

        if(vec == null)
            return new ArrayList<>();
        Counter<String> distances = new Counter<>();

        for(String s : cache.words()) {
            if(s.equals(word))
                continue;
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(vec,otherVec);
            distances.incrementCount(s, sim);
        }


        distances.keepTopNKeys(n);
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
        double min = Double.MAX_VALUE;
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
    public void fit() {
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            cache.saveVocab();
        if (stopWords == null)
            readStopWords();


        log.info("Training word2vec multithreaded");

        if (sentenceIter != null)
            sentenceIter.reset();
        if (docIter != null)
            docIter.reset();


        final ExecutorService service = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2);
        final Collection<Integer> docs = vectorizer.index().allDocs();
        int tries = 0;
        while(docs.isEmpty()) {
            if(tries >= 3)
                throw new IllegalStateException("Unable to train, no documents found");
            else {
                log.warn("No documents found...waiting 10 seconds on try " + tries);
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                tries++;
            }
        }

        final AtomicInteger numSentencesProcessed = new AtomicInteger(0);
        totalWords = vectorizer.numWordsEncountered();
        totalWords *= numIterations;


        log.info("Processing sentences...");

        final List<Future<?>> futures2 = new ArrayList<>();
        for (int i = 0; i < numIterations; i++) {
            log.info("Training on " + docs.size());
            final AtomicLong nextRandom = new AtomicLong(5);


            Iterator<List<VocabWord>> minibatchesIter = vectorizer.index().miniBatches();
            while(minibatchesIter.hasNext()) {
                final List<VocabWord> batch = minibatchesIter.next();
                futures2.add(service.submit(new Callable<Void>() {


                    @Override
                    public Void call() {
                        trainSentence(batch, numSentencesProcessed,nextRandom);

                        return null;
                    }
                }));
            }
        }





        try {

            for(Future<?> f : futures2)
                f.get();
            service.shutdown();
            while(!service.isTerminated())
                Thread.sleep(1000);

        } catch (Exception e) {
            Thread.currentThread().interrupt();
        }


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
    public boolean buildVocab() {
        readStopWords();

        if(cache.vocabExists()) {
            log.info("Loading vocab...");
            cache.loadVocab();
            cache.resetWeights();
            return true;
        }

        //vectorizer will handle setting up vocab meta data
        if(vectorizer == null)
            vectorizer = new TfidfVectorizer.Builder()
                    .cache(cache).iterate(docIter).iterate(sentenceIter).batchSize(batchSize)
                    .minWords(minWordFrequency).stopWords(stopWords)
                    .tokenize(tokenizerFactory).build();
        vectorizer.fit();

        setup();

        return false;
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
    public void trainSentence(final List<VocabWord> sentence,AtomicInteger numWordsSoFar,AtomicLong nextRandom) {
        if(sentence == null || sentence.isEmpty())
            return;

        numWordsSoFar.set(numWordsSoFar.get() + sentence.size());
        rateOfChange.set(rateOfChange.get() + sentence.size());
        if(rateOfChange.get() >=  learningRateDecayWords) {
            rateOfChange.set(0);
            //use learning rate decay instead
            if(!useAdaGrad) {
                alpha.set(Math.max(minLearningRate, alpha.get() * (1 - (1.0 * (double) numWordsSoFar.get() / (double) totalWords))));
                cache.setLearningRate(alpha.get());
            }
            log.info("Num words so far " + numWordsSoFar.get() + " alpha is " + alpha.get() + " out of " + totalWords);
        }




        for(int i = 0; i < sentence.size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            skipGram(i, sentence, (int) nextRandom.get() % window,nextRandom);
        }
    }


    /**
     * Train via skip gram
     * @param i
     * @param sentence
     */
    public void skipGram(int i,List<VocabWord> sentence, int b,AtomicLong nextRandom) {
        final VocabWord word = sentence.get(i);
        if(word == null || sentence.isEmpty())
            return;

        int end =  window * 2 + 1 - b;

        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    VocabWord lastWord = sentence.get(c);
                    iterate(word,lastWord,nextRandom);
                }
            }
        }

    }

    /**
     * Train the word vector
     * on the given words
     * @param w1 the first word to fit
     */
    public void  iterate(VocabWord w1, VocabWord w2,AtomicLong nextRandom) {
        cache.iterateSample(w1,w2,nextRandom);

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

        INDArray vector = Transforms.unitVec(getWordVectorMatrix(word));
        INDArray vector2 = Transforms.unitVec(getWordVectorMatrix(word2));
        if(vector == null || vector2 == null)
            return -1;
        return  Nd4j.getBlasWrapper().dot(vector,vector2);
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
            this.alpha = vec.alpha;
            this.minWordFrequency = vec.minWordFrequency;
            this.sample = vec.sample;
            this.stopWords = vec.stopWords;
            this.topNSize = vec.topNSize;
            this.window = vec.window;

        }catch(Exception e) {
            throw new RuntimeException(e);
        }



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


    /**
     * restart training on next fit().
     * Use when sentence iterator is set for new training.
     */
    public void resetWeightsOnSetup() {
        this.shouldReset = true;
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


    public static class Builder {
        private int minWordFrequency = 1;
        private int layerSize = 50;
        private SentenceIterator iter;
        private List<String> stopWords = StopWords.getStopWords();
        private int window = 5;
        private TokenizerFactory tokenizerFactory;
        private VocabCache vocabCache;
        private DocumentIterator docIter;
        private double lr = 2.5e-1;
        private int iterations = 1;
        private long seed = 123;
        private boolean saveVocab = false;
        private int batchSize = 1000;
        private int learningRateDecayWords = 10000;
        private boolean useAdaGrad = false;
        private TextVectorizer textVectorizer;
        private double minLearningRate = 1e-2;
        private double negative = 0;
        private double sampling = 1e-5;

        public Builder sampling(double sample) {
            this.sampling = sample;
            return this;
        }


        public Builder negativeSample(double negative) {
            this.negative = negative;
            return this;
        }

        public Builder minLearningRate(double minLearningRate) {
            this.minLearningRate = minLearningRate;
            return this;
        }


        public Builder useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder vectorizer(TextVectorizer textVectorizer) {
            this.textVectorizer = textVectorizer;
            return this;
        }

        public Builder learningRateDecayWords(int learningRateDecayWords) {
            this.learningRateDecayWords = learningRateDecayWords;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Builder saveVocab(boolean saveVocab){
            this.saveVocab = saveVocab;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }


        public Builder learningRate(double lr) {
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
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.setCache(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.sample = sampling;


                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr)
                            .vectorLength(layerSize).build();

                    ret.cache = vocabCache;
                }
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
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.vectorizer = textVectorizer;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setCache(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr)
                            .vectorLength(layerSize).build();
                    ret.cache = vocabCache;
                }
                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }



        }
    }




}