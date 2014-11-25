package org.deeplearning4j.models.word2vec;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;


import com.google.common.base.Function;
import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.util.SetUtils;
import org.eclipse.jetty.util.ConcurrentHashSet;
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


    protected static final long serialVersionUID = -2367495638286018038L;

    protected transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    protected transient SentenceIterator sentenceIter;
    protected transient DocumentIterator docIter;
    protected transient VocabCache cache;
    protected int batchSize = 1000;
    protected int topNSize = 40;
    protected double sample = 0;
    protected long totalWords = 1;
    protected AtomicInteger rateOfChange = new AtomicInteger(0);
    //learning rate
    protected AtomicDouble alpha = new AtomicDouble(0.025);
    //number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
    protected int minWordFrequency = 5;
    //context to use for gathering word frequencies
    protected int window = 5;
    //number of neurons per layer
    protected int layerSize = 50;
    protected transient  RandomGenerator g;
    protected static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    protected List<String> stopWords;
    protected boolean shouldReset = true;
    //number of iterations to run
    protected int numIterations = 1;
    public final static String UNK = "UNK";
    protected long seed = 123;
    protected boolean saveVocab = false;
    protected double minLearningRate = 0.01;
    protected TextVectorizer vectorizer;
    protected int learningRateDecayWords = 10000;
    protected boolean useAdaGrad = false;
    protected LinkedBlockingDeque<List<VocabWord>> jobQueue = new LinkedBlockingDeque<>(100000);
    protected AtomicLong timeLastUpdated = new AtomicLong(0);

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
    public Collection<String> wordsNearestSum(List<String> positive,List<String> negative,int top) {
        INDArray words = Nd4j.create(layerSize);
        Set<String> union = SetUtils.union(new HashSet<>(positive),new HashSet<>(negative));
        for(String s : positive)
            words.addi(cache.vector(s));


        for(String s : negative)
            words.addi(cache.vector(s).mul(-1));


        if(cache instanceof  InMemoryLookupCache) {
            InMemoryLookupCache l = (InMemoryLookupCache) cache;
            INDArray syn0 = l.getSyn0();
            INDArray weights = syn0.norm2(0).rdivi(1).muli(words);
            INDArray distances = syn0.mulRowVector(weights).sum(1);
            INDArray[] sorted = Nd4j.sortWithIndices(distances,0,false);
            INDArray sort = sorted[0];
            List<String> ret = new ArrayList<>();
            if(top > sort.length())
                top = sort.length();
            //there will be a redundant word
            int end = top + 1;
            for(int i = 0; i < end; i++) {
                String word = cache.wordAtIndex(sort.getInt(i));
                if(union.contains(word)) {
                    end++;
                    if(end >= sort.length())
                        break;
                    continue;
                }
                ret.add(cache.wordAtIndex(sort.getInt(i)));
            }


            return ret;
        }

        Counter<String> distances = new Counter<>();

        for(String s : cache.words()) {
            INDArray otherVec = getWordVectorMatrix(s);
            double sim = Transforms.cosineSim(words,otherVec);
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
    public Collection<String> wordsNearestSum(String word,int n) {
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
     * Words nearest based on positive and negative words
     * @param positive the positive words
     * @param negative the negative words
     * @param top the top n words
     * @return the words nearest the mean of the words
     */
    public Collection<String> wordsNearest(List<String> positive,List<String> negative,int top) {
        INDArray words = Nd4j.create(positive.size() + negative.size(),layerSize);
        int row = 0;
        Set<String> union = SetUtils.union(new HashSet<>(positive),new HashSet<>(negative));
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
            int end = top + 1;
            for(int i = 0; i < end; i++) {
                String word = cache.wordAtIndex(sort.getInt(i));
                if(union.contains(word)) {
                    end++;
                    if(end >= sort.length())
                        break;
                    continue;
                }
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
    public void fit() throws IOException {
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


        final int[] docs = vectorizer.index().allDocs();

        final AtomicLong numSentencesProcessed = new AtomicLong(0);
        totalWords = vectorizer.numWordsEncountered();
        totalWords *= numIterations;



        log.info("Processing sentences...");


        List<Thread> work = new ArrayList<>();
        final AtomicInteger processed = new AtomicInteger(0);
        for(int i = 0; i < Runtime.getRuntime().availableProcessors(); i++) {
            final Set<List<VocabWord>> set = new ConcurrentHashSet<>();

            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    final AtomicLong nextRandom = new AtomicLong(5);
                    while(true) {
                        if(processed.get() >= docs.length)
                            break;
                        List<VocabWord> job = jobQueue.poll();
                        if(job == null || job.isEmpty() || set.contains(job))
                            continue;
                        if(set.contains(job))
                            continue;

                        set.add(job);
                        trainSentence(job, numSentencesProcessed, nextRandom);
                        processed.incrementAndGet();
                        log.info("Ran " + processed.get() + " so far");



                    }
                }
            });

            t.setName("worker" + i);
            t.start();
            work.add(t);
        }


        final AtomicLong nextRandom = new AtomicLong(5);
        final AtomicInteger doc = new AtomicInteger(0);
        final int numDocs = vectorizer.index().numDocuments() * numIterations;
        ExecutorService exec = new ThreadPoolExecutor(Runtime.getRuntime().availableProcessors(),
                Runtime.getRuntime().availableProcessors(),
                0L, TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<Runnable>(), new RejectedExecutionHandler() {
            @Override
            public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                executor.submit(r);
            }
        });


        vectorizer.index().eachDoc(new Function<List<VocabWord>, Void>() {
            @Override
            public Void apply(List<VocabWord> input) {
                List<VocabWord> batch = new ArrayList<>();
                addWords(input, nextRandom, batch);
                if(batch.isEmpty())
                    return null;

                try {
                    for(int i = 0; i < numIterations; i++)
                        while(!jobQueue.offer(batch,1,TimeUnit.MILLISECONDS))
                            Thread.sleep(1);



                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }

                doc.incrementAndGet();
                if(doc.get() > 0 && doc.get() % 10000 == 0)
                    log.info("Doc " + doc.get() + " done so far out of " + numDocs);

                return null;
            }
        },exec);

        exec.shutdown();
        try {
            exec.awaitTermination(1,TimeUnit.DAYS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        for(Thread t : work)
            try {
                t.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }


    }



    protected void addWords(List<VocabWord> sentence,AtomicLong nextRandom,List<VocabWord> currMiniBatch) {
        for (VocabWord word : sentence) {
            if(word == null)
                continue;
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0) {
                double numDocs =  vectorizer.index().numDocuments();
                double ran = (Math.sqrt(word.getWordFrequency() / (sample * numDocs)) + 1)
                        * (sample * numDocs) / word.getWordFrequency();

                if (ran < (nextRandom.get() & 0xFFFF) / (double) 65536) {
                    continue;
                }

                currMiniBatch.add(word);
            }
            else
                currMiniBatch.add(word);



        }



    }


    /**
     * Build the binary tree
     * Reset the weights
     */
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
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(final List<VocabWord> sentence,AtomicLong numWordsSoFar,AtomicLong nextRandom) {
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
    protected void buildBinaryTree() {
        log.info("Constructing priority queue");
        Huffman huffman = new Huffman(cache.vocabWords());
        huffman.build();

        log.info("Built tree");

    }




    /* reinit weights */
    protected void resetWeights() {
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
    protected void readStopWords() {
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
        if(cache instanceof InMemoryLookupCache) {
            InMemoryLookupCache l = (InMemoryLookupCache) cache;
            if(l.getSyn0() != null && l.getSyn0().columns() != layerSize)
                layerSize = l.getSyn0().columns();
        }
    }


    public static class Builder {
        protected int minWordFrequency = 1;
        protected int layerSize = 50;
        protected SentenceIterator iter;
        protected List<String> stopWords = StopWords.getStopWords();
        protected int window = 5;
        protected TokenizerFactory tokenizerFactory;
        protected VocabCache vocabCache;
        protected DocumentIterator docIter;
        protected double lr = 2.5e-1;
        protected int iterations = 1;
        protected long seed = 123;
        protected boolean saveVocab = false;
        protected int batchSize = 1000;
        protected int learningRateDecayWords = 10000;
        protected boolean useAdaGrad = false;
        protected TextVectorizer textVectorizer;
        protected double minLearningRate = 1e-2;
        protected double negative = 0;
        protected double sampling = 1e-5;

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