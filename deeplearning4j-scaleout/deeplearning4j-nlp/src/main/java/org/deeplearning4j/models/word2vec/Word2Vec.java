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
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.eclipse.jetty.util.ConcurrentHashSet;
import org.deeplearning4j.nn.api.Persistable;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
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
public class Word2Vec extends WordVectorsImpl implements Persistable {


    protected static final long serialVersionUID = -2367495638286018038L;

    protected transient TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    protected transient SentenceIterator sentenceIter;
    protected transient DocumentIterator docIter;
    protected int batchSize = 1000;
    protected double sample = 0;
    protected long totalWords = 1;
    protected AtomicInteger rateOfChange = new AtomicInteger(0);
    //learning rate
    protected AtomicDouble alpha = new AtomicDouble(0.025);
    //number of times the word must occur in the vocab to appear in the calculations, otherwise treat as unknown
    protected int minWordFrequency = 5;
    //context to use for gathering word frequencies
    protected int window = 5;
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
    protected InvertedIndex invertedIndex;
    protected boolean useAdaGrad = false;
    protected WeightLookupTable lookupTable;
    protected int workers = Runtime.getRuntime().availableProcessors();
    protected Queue<List<List<VocabWord>>> jobQueue = new LinkedBlockingDeque<>(10000);

    public Word2Vec() {}







    public TextVectorizer getVectorizer() {
        return vectorizer;
    }

    public void setVectorizer(TextVectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }

   

    /**
     * Train the model
     */
    public void fit() throws IOException {
        boolean loaded = buildVocab();
        //save vocab after building
        if (!loaded && saveVocab)
            vocab().saveVocab();
        if (stopWords == null)
            readStopWords();


        log.info("Training word2vec multithreaded");

        if (sentenceIter != null)
            sentenceIter.reset();
        if (docIter != null)
            docIter.reset();


        final int[] docs = vectorizer.index().allDocs();

        totalWords = vectorizer.numWordsEncountered();
        if(totalWords < 1)
            throw new IllegalStateException("Unable to train, total words less than 1");

        totalWords *= numIterations;



        log.info("Processing sentences...");


        List<Thread> work = new ArrayList<>();
        final AtomicInteger processed = new AtomicInteger(0);
        final int allDocs = docs.length * numIterations;
        final AtomicLong numWordsSoFar = new AtomicLong(0);
        final AtomicLong lastReport = new AtomicLong(0);
        for(int i = 0; i < workers; i++) {
            final Set<List<VocabWord>> set = new ConcurrentHashSet<>();

            Thread t = new Thread(new Runnable() {
                @Override
                public void run() {
                    final AtomicLong nextRandom = new AtomicLong(5);
                    long checked = 0;
                    while(true) {
                        if(checked > 0 && checked % 1000 == 0 && processed.get() >= allDocs)
                            return;
                        checked++;
                        List<List<VocabWord>> job = jobQueue.poll();
                        if(job == null || job.isEmpty() || set.contains(job))
                            continue;

                        double alpha = Math.max(minLearningRate, Word2Vec.this.alpha.get() * (1 - (1.0 * (double) numWordsSoFar.get() / (double) totalWords)));
                        long diff = Math.abs(lastReport.get() - numWordsSoFar.get());
                        if(numWordsSoFar.get() > 0 && diff >=  10000) {
                            log.info("Words so far " + numWordsSoFar.get() + " with alpha at " + alpha);
                            lastReport.set(numWordsSoFar.get());
                        }
                        long increment = 0;
                        for(List<VocabWord> sentence : job) {
                            trainSentence(sentence, nextRandom, alpha);
                            increment += sentence.size();
                        }

                        numWordsSoFar.set(numWordsSoFar.get() + increment);
                        processed.set(processed.get() + job.size());



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

        final Queue<List<VocabWord>> batch2 = new ConcurrentLinkedDeque<>();
        vectorizer.index().eachDoc(new Function<List<VocabWord>, Void>() {
            @Override
            public Void apply(List<VocabWord> input) {
                List<VocabWord> batch = new ArrayList<>();
                addWords(input, nextRandom, batch);
                if(batch.isEmpty())
                    return null;

                for(int i = 0; i < numIterations; i++) {
                    batch2.add(batch);
                }

                if(batch2.size() >= 100 || batch2.size() >= numDocs) {
                    boolean added = false;
                    while(!added) {
                        try {
                            jobQueue.add(new LinkedList<>(batch2));
                            batch2.clear();
                            added = true;
                        }catch(Exception e) {
                            continue;
                        }
                    }

                }


                doc.incrementAndGet();
                if(doc.get() > 0 && doc.get() % 10000 == 0)
                    log.info("Doc " + doc.get() + " done so far");

                return null;
            }
        },exec);

        if(!batch2.isEmpty())
            jobQueue.add(new LinkedList<>(batch2));




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

        if(vocab().vocabExists()) {
            log.info("Loading vocab...");
            vocab().loadVocab();
            lookupTable.resetWeights();
            return true;
        }

        //vectorizer will handle setting up vocab meta data
        if(vectorizer == null) {

            if(invertedIndex == null)
                invertedIndex = new LuceneInvertedIndex.Builder()
                        .cache(vocab()).stopWords(stopWords)
                        .build();

            vectorizer = new TfidfVectorizer.Builder().index(invertedIndex)
                    .cache(vocab()).iterate(docIter).iterate(sentenceIter).batchSize(batchSize)
                    .minWords(minWordFrequency).stopWords(stopWords)
                    .tokenize(tokenizerFactory).build();
        }

        vectorizer.fit();

        setup();

        return false;
    }



    /**
     * Train on a list of vocab words
     * @param sentence the list of vocab words to train on
     */
    public void trainSentence(final List<VocabWord> sentence,AtomicLong nextRandom,double alpha) {
        if(sentence == null || sentence.isEmpty())
            return;
        for(int i = 0; i < sentence.size(); i++) {
            nextRandom.set(nextRandom.get() * 25214903917L + 11);
            skipGram(i, sentence, (int) nextRandom.get() % window,nextRandom,alpha);
        }

    }


    /**
     * Train via skip gram
     * @param i
     * @param sentence
     */
    public void skipGram(int i,List<VocabWord> sentence, int b,AtomicLong nextRandom,double alpha) {

        final VocabWord word = sentence.get(i);
        if(word == null || sentence.isEmpty())
            return;

        int end =  window * 2 + 1 - b;
        for(int a = b; a < end; a++) {
            if(a != window) {
                int c = i - window + a;
                if(c >= 0 && c < sentence.size()) {
                    VocabWord lastWord = sentence.get(c);
                    iterate(word,lastWord,nextRandom,alpha);
                }
            }
        }




    }

    /**
     * Train the word vector
     * on the given words
     * @param w1 the first word to fit
     */
    public void  iterate(VocabWord w1, VocabWord w2,AtomicLong nextRandom,double alpha) {
        lookupTable.iterateSample(w1,w2,nextRandom,alpha);

    }




    /* Builds the binary tree for the word relationships */
    protected void buildBinaryTree() {
        log.info("Constructing priority queue");
        Huffman huffman = new Huffman(vocab().vocabWords());
        huffman.build();

        log.info("Built tree");

    }




    /* reinit weights */
    protected void resetWeights() {
        lookupTable.resetWeights();
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
            this.window = vec.window;

        }catch(Exception e) {
            throw new RuntimeException(e);
        }



    }

    public WeightLookupTable getLookupTable() {
        return lookupTable;
    }

    public void setLookupTable(WeightLookupTable lookupTable) {
        this.lookupTable = lookupTable;
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



    public void setCache(WeightLookupTable lookupTable) {
        this.lookupTable = lookupTable;
        if(vocab() instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) vocab();
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
        protected int workers = Runtime.getRuntime().availableProcessors();
        protected InvertedIndex index;
        protected WeightLookupTable lookupTable;

        public Builder lookupTable(WeightLookupTable lookupTable) {
            this.lookupTable = lookupTable;
            return this;
        }

        public Builder index(InvertedIndex index) {
            this.index = index;
            return this;
        }

        public Builder workers(int workers) {
            this.workers = workers;
            return this;
        }

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
                ret.setVocab(vocabCache);
                ret.numIterations = iterations;
                ret.minWordFrequency = minWordFrequency;
                ret.seed = seed;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.useAdaGrad = useAdaGrad;
                ret.minLearningRate = minLearningRate;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;
                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }


                ret.docIter = docIter;
                ret.lookupTable = lookupTable;
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
                ret.setVocab(vocabCache);
                ret.docIter = docIter;
                ret.minWordFrequency = minWordFrequency;
                ret.numIterations = iterations;
                ret.seed = seed;
                ret.numIterations = iterations;
                ret.saveVocab = saveVocab;
                ret.batchSize = batchSize;
                ret.sample = sampling;
                ret.workers = workers;
                ret.invertedIndex = index;
                ret.lookupTable = lookupTable;

                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }

                if(vocabCache == null) {
                    vocabCache = new InMemoryLookupCache();

                    ret.setVocab(vocabCache);
                }

                if(lookupTable == null) {
                    lookupTable = new InMemoryLookupTable.Builder().negative(negative)
                            .useAdaGrad(useAdaGrad).lr(lr).cache(vocabCache)
                            .vectorLength(layerSize).build();
                }
                ret.lookupTable = lookupTable;
                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }



        }
    }




}