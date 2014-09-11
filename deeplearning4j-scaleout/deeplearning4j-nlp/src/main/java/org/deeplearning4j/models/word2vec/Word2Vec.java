package org.deeplearning4j.models.word2vec;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Triple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.nn.api.Persistable;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.models.word2vec.actor.SentenceActor;
import org.deeplearning4j.models.word2vec.actor.VocabActor;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.ehcache.EhCacheVocabCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;
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
    private transient VocabCache cache = new EhCacheVocabCache();
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
    private transient  RandomGenerator g = new MersenneTwister(123);
    private static Logger log = LoggerFactory.getLogger(Word2Vec.class);
    private int size = 0;
    private int words = 0;
    private int allWordsCount = 0;
    private AtomicInteger numSentencesProcessed = new AtomicInteger(0);
    private static ActorSystem trainingSystem;
    private List<String> stopWords;
    /* out of vocab */
    private float[] oob;
    private boolean shouldReset = true;

    public Word2Vec() {}

    /**
     * Specify a sentence iterator
     *
     *
     *
     *
     *
     */
    public Word2Vec(SentenceIterator sentenceIter) {
        oob = new float[layerSize];
        Arrays.fill(oob, 0.0f);
        readStopWords();
        this.sentenceIter = sentenceIter;
    }



    public Word2Vec(SentenceIterator sentenceIter,int minWordFrequency) {
        oob = new float[layerSize];
        Arrays.fill(oob,0.0f);
        readStopWords();
        this.sentenceIter = sentenceIter;
        this.minWordFrequency = minWordFrequency;
    }


    public Word2Vec(TokenizerFactory factory,SentenceIterator sentenceIter) {
        this(sentenceIter);
        this.tokenizerFactory = factory;
    }

    /**
     * Specify a custom tokenizer, sentence iterator
     * and minimum word frequency
     * @param factory
     * @param sentenceIter
     * @param minWordFrequency
     */
    public Word2Vec(TokenizerFactory factory,SentenceIterator sentenceIter,int minWordFrequency) {
        this(factory,sentenceIter);
        this.minWordFrequency = minWordFrequency;
    }



    /**
     * Assumes whole dataset is passed in.
     * This is purely meant for batch methods.
     * Same as calling {@link #Word2Vec(Collection, int)}
     * with the second argument being 5
     * @param sentences the sentences to use
     * to iterate on
     */
    public Word2Vec(Collection<String> sentences) {
        this(sentences,5);
        readStopWords();

    }

    public Word2Vec(Collection<String> sentences,TokenizerFactory factory) {
        this(sentences);
        this.tokenizerFactory = factory;
    }

    /**
     * Initializes based on assumption of whole data applyTransformToDestination being passed in.
     * @param sentences the sentences to be used for training
     * @param minWordFrequency the minimum word frequency
     * to be counted in the vocab
     */
    public Word2Vec(Collection<String> sentences,int minWordFrequency) {
        this.minWordFrequency = minWordFrequency;
        this.sentenceIter = new CollectionSentenceIterator(sentences);

        this.buildVocab();
        oob = new float[layerSize];
        Arrays.fill(oob,0.0f);
        readStopWords();

    }


    public Word2Vec(Collection<String> sentences,int minWordFrequency,TokenizerFactory factory) {
        this(sentences,minWordFrequency);
        this.tokenizerFactory = factory;
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
        for(int i = 0; i < cache.numWords(); i++) {
            if(MathUtils.stringSimilarity(word,cache.wordAtIndex(i)) >= accuracy)
                ret.add(cache.wordAtIndex(i));
        }
        return ret;
    }




    public int indexOf(String word) {
        return cache.indexOf(word);
    }

    public INDArray getWordVectorMatrix(String word) {
        int i = this.cache.indexOf(word);
        if(oob == null)
            oob = new float[layerSize];

        if(i < 0)
            return Nd4j.create(oob, new int[]{layerSize});
        return cache.vector(word);
    }

    public INDArray getWordVectorMatrixNormalized(String word) {
        int i = this.cache.indexOf(word);
        if(oob == null)
            oob = new float[layerSize];
        if(i < 0)
            return Nd4j.zeros(layerSize);
        INDArray r =  cache.vector(word);
        return r.div((Nd4j.getBlasWrapper().nrm2(r)));
    }








    public Collection<String> wordsNearest(String word,int n) {
        INDArray vec = this.getWordVectorMatrix(word);
        if(vec == null)
            return new ArrayList<>();
        Counter<String> distances = new Counter<String>();
        for(int i = 0; i <  cache.numWords(); i++) {
            double sim = similarity(word,cache.wordAtIndex(i).toString());
            distances.incrementCount(cache.wordAtIndex(i).toString(), sim);
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
            VocabWord v = new VocabWord(score,layerSize);
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
            VocabWord w = new VocabWord(score,layerSize);
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
        if(trainingSystem == null) {
            trainingSystem = ActorSystem.create();

        }


        buildVocab();

        if(stopWords == null)
            readStopWords();


        log.info("Training word2vec multithreaded");





        final AtomicLong changed = new AtomicLong(System.currentTimeMillis());


        trainingSystem.actorOf(new RoundRobinPool(Runtime.getRuntime().availableProcessors() *3 ).props(Props.create(new SentenceActor.SentenceActorCreator(this)).withDispatcher("akka.actor.worker-dispatcher")));


        int numCores = Runtime.getRuntime().availableProcessors();

        final CountDownLatch latch = new CountDownLatch(numCores);


        for(int i = 0; i < numCores; i++) {
            Future<Void> f = Futures.future(new Callable<Void>() {
                @Override
                public Void call() throws Exception {
                    while(getSentenceIter().hasNext()) {

                        final String sentence = sentenceIter.nextSentence();
                        if(sentence != null && !sentence.isEmpty()) {
                            Future<Void> f = Futures.future(new Callable<Void>() {

                                @Override
                                public Void call() throws Exception {
                                    processSentence(sentence);
                                    return null;
                                }

                            },trainingSystem.dispatcher());
                            f.onComplete(new OnComplete<Void>() {

                                @Override
                                public void onComplete(Throwable arg0, Void arg1)
                                        throws Throwable {
                                    if(arg0 != null)
                                        throw arg0;
                                    numSentencesProcessed.incrementAndGet();
                                    changed.set(System.currentTimeMillis());

                                }

                            }, trainingSystem.dispatcher());

                        }

                    }
                    return null;
                }
            },trainingSystem.dispatcher());
            f.onComplete(new OnComplete<Void>() {
                @Override
                public void onComplete(Throwable failure, Void success) throws Throwable {
                    latch.countDown();
                }
            },trainingSystem.dispatcher());
        }


        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();

        }


        boolean done = false;
        long fiveMinutes = TimeUnit.MINUTES.toMillis(1);
        while(!done) {
            long curr = System.currentTimeMillis();
            long lastChanged = changed.get();
            long diff = Math.abs(curr - lastChanged);
            //hasn't changed for 5 minutes
            if(diff >= fiveMinutes) {
                done = true;
            }

            else
                try {
                    Thread.sleep(15000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
        }



        if(trainingSystem != null)
            trainingSystem.shutdown();

    }

    public void processSentence(final String sentence) {
        trainSentence(sentence);
        if(numSentencesProcessed.get() % 10000 == 0) {
            float newAlpha =  alpha.floatValue() * (1 -  (float) cache.totalWordOccurrences() / allWordsCount);
            float oldAlpha = alpha.floatValue();
            if(Float.isNaN(newAlpha))
                newAlpha = oldAlpha;
            alpha.set(Math.max(MIN_ALPHA,newAlpha));
            log.info("Alpha updated " + alpha + " progress " + numSentencesProcessed);
        }
    }


    /**
     * Train on the given sentence returning a list of vocab words
     * @param sentence the sentence to fit on
     * @return
     */
    public List<VocabWord> trainSentence(String sentence) {
        Tokenizer tokenizer = tokenizerFactory.create(sentence);
        List<VocabWord> sentence2 = new ArrayList<>();
        while(tokenizer.hasMoreTokens()) {
            String next = tokenizer.nextToken();
            if(stopWords.contains(next))
                next = "STOP";
            VocabWord word = cache.wordFor(next);
            if(word == null)
                continue;

            sentence2.add(word);
            cache.incrementWordCount(next);

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
        String name;
        for (int i = 0; i < cache.numWords(); i++) {
            name = cache.wordAtIndex(i);
            if (name.equals(word)) {
                continue;
            }

            tempVector = cache.vector(cache.wordAtIndex(i));
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

        if(trainingSystem == null)
            trainingSystem = ActorSystem.create();

        final AtomicLong semaphore = new AtomicLong(System.currentTimeMillis());
        final AtomicInteger queued = new AtomicInteger(0);

        final ActorRef vocabActor = trainingSystem.actorOf(
                new RoundRobinPool(Runtime.getRuntime().availableProcessors()).props(
                        Props.create(VocabActor.class,tokenizerFactory,cache,layerSize,stopWords,semaphore)));

		/* all words; including those not in the actual ending index */
        int numCores = Runtime.getRuntime().availableProcessors();

        final CountDownLatch latch = new CountDownLatch(numCores);

        for(int i = 0; i < numCores; i++) {

            Future<Void> f =  Futures.future(new Callable<Void>() {
                @Override
                public Void call() throws Exception {
                    while(getSentenceIter().hasNext()) {
                        String sentence = getSentenceIter().nextSentence();
                        if(sentence == null)
                            return null;

                        vocabActor.tell(sentence, vocabActor);
                        queued.incrementAndGet();
                        if(queued.get() % 10000 == 0)
                            log.info("Sent " + queued);


                    }

                    return null;
                }
            },trainingSystem.dispatcher());


            f.onComplete(new OnComplete<Void>() {
                @Override
                public void onComplete(Throwable failure, Void success) throws Throwable {
                   if(failure != null)
                       log.error("Error thrown ",failure);
                    latch.countDown();
                }
            },trainingSystem.dispatcher());

        }


        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }


        boolean done = false;
        long fiveMinutes = TimeUnit.MINUTES.toMillis(1);
        while(!done) {
            long curr = System.currentTimeMillis();
            long lastChanged = semaphore.get();
            long diff = Math.abs(curr - lastChanged);
            log.info("Waiting on setup...");
            //hasn't changed for 5 minutes
            if(diff >= fiveMinutes) {
                done = true;
            }

            else
                try {
                    Thread.sleep(15000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
        }

        setup();

    }

    public void trainSentence(List<VocabWord> sentence) {
        long nextRandom = 5;
        for(int i = 0; i < sentence.size(); i++) {
            VocabWord entry = sentence.get(i);
            // The subsampling randomly discards frequent words while keeping the ranking same
            if (sample > 0) {
                double ran = (Math.sqrt(entry.getWordFrequency() / (sample * trainWordsCount)) + 1)
                        * (sample * trainWordsCount) / entry.getWordFrequency();
                nextRandom = nextRandom * 25214903917L + 11;
                if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                    continue;
                }
            }

            nextRandom = nextRandom * 25214903917L + 11;
            int b = (int) nextRandom % window;
            skipGram(i,sentence,b);
        }
    }


    /**
     * Train via skip gram
     * @param i
     * @param sentence
     * @param b
     */
    public void skipGram(int i,List<VocabWord> sentence,int b) {
        VocabWord word = sentence.get(i);
        if(word == null)
            return;


        //subsampling
        for(int j = b; j < window * 2 + 1 - b; j++) {
            if(j == window)
                continue;
            int c1 = i - window + j;

            if (c1 < 0 || c1 >= sentence.size())
                continue;

            VocabWord word2 = sentence.get(c1);
            iterate(word);
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
    public void  iterate(VocabWord w1) {
        INDArray l1 = cache.vector(cache.wordAtIndex(w1.getIndex()));
        INDArray l2a = cache.loadCodes(w1.getCodes());
        INDArray fa = Transforms.sigmoid(l1.mmul(l2a.transpose()));
        // ga = (1 - word.code - fa) * alpha  # vector of error gradients multiplied by the learning rate
        INDArray ga = Nd4j.ones(fa.length()).subi(ArrayUtil.toNDArray(w1.getCodes())).subi(fa).muli(alpha.floatValue());
        INDArray outer = ga.mmul(l1);
        for(int i = 0; i < w1.getPoints().length; i++) {
            INDArray toAdd = l2a.getRow(i).addi(outer.getRow(i));
            cache.putCode(w1.getPoints()[i], toAdd);
        }

        cache.putVector(cache.wordAtIndex(w1.getIndex()), l1.addi(ga.mmul(l2a)));
    }




    /* Builds the binary tree for the word relationships */
    private void buildBinaryTree() {
        log.info("Constructing priority queue");
        PriorityQueue<VocabWord> heap = new PriorityQueue<>(cache.vocabWords());
        int i = 0;
        int heapCount = 0;

        log.info("Beginning tree construction");
        while(heap.size() > 1) {
            VocabWord min1 = heap.poll();
            VocabWord min2 = heap.poll();

            if(heapCount % 1000 == 0) {
                log.info("Heap progress o far " + heapCount);
            }
            VocabWord add = new VocabWord(min1.getWordFrequency() + min2.getWordFrequency(),layerSize);
            int index = (cache.numWords() + i);

            add.setIndex(index);
            add.setLeft(min1);
            add.setRight(min2);
            min1.setCode(0);
            min2.setCode(1);
            min1.setParent(add);
            min2.setParent(add);
            heap.add(add);
            i++;
            heapCount++;
        }

        Triple<VocabWord,int[],int[]> triple = new Triple<>(heap.poll(),new int[]{},new int[]{});
        Stack<Triple<VocabWord,int[],int[]>> stack = new Stack<>();
        log.info("Beginning stack operation");

        stack.add(triple);

        int stackCount = 0;

        while(!stack.isEmpty())  {
            if(stackCount % 1000 == 0) {
                log.info("Stack count so far" + stackCount);
            }
            triple = stack.pop();
            int[] codes = triple.getSecond();
            int[] points = triple.getThird();
            VocabWord node = triple.getFirst();
            if(node == null) {
                log.info("Node was null");
                continue;
            }
            if(node.getIndex() < cache.numWords()) {
                node.setCodes(codes);
                node.setPoints(points);
            }
            else {
                int[] copy = plus(points,node.getIndex() - cache.numWords());
                points = copy;
                triple.setThird(points);
                stack.add(new Triple<>(node.getLeft(),plus(codes,0),points));
                stack.add(new Triple<>(node.getRight(),plus(codes,1),points));

            }
            stackCount++;
        }



        log.info("Built tree");
    }

    private int[] plus (int[] addTo,int add) {
        int[] copy = new int[addTo.length + 1];
        for(int c = 0; c < addTo.length; c++)
            copy[c] = addTo[c];
        copy[addTo.length] = add;
        return copy;
    }


    /* reinit weights */
    private void resetWeights() {
        for(int i = 0; i < cache.numWords(); i++)
            cache.putVector(cache.wordAtIndex(i),randomVector());



    }

    public INDArray randomVector() {
        INDArray ret = Nd4j.create(layerSize);
        for(int i = 0; i < ret.length(); i++) {
            ret.putScalar(i, g.nextFloat() - 0.5f / layerSize);
        }
        return ret;
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
        INDArray vector = getWordVectorMatrixNormalized(word);
        INDArray vector2 = getWordVectorMatrixNormalized(word2);
        if(vector == null || vector2 == null)
            return -1;
        INDArray d1 = Transforms.unitVec(vector);
        INDArray d2 = Transforms.unitVec(vector2);
        double ret = Nd4j.getBlasWrapper().dot(d1, d2);
        if(ret <  0)
            return 0;
        return ret;
    }




    @SuppressWarnings("unchecked")
    private void readStopWords() {
        stopWords = StopWords.getStopWords();


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





    public float[]	 getOob() {
        if(oob == null)
            oob = new float[layerSize];
        return oob;
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
            this.oob = vec.oob;
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


    public static class Builder {
        private int minWordFrequency = 5;
        private int layerSize = 50;
        private SentenceIterator iter;
        private List<String> stopWords = StopWords.getStopWords();
        private int window = 5;
        private TokenizerFactory tokenizerFactory;
        private VocabCache vocabCache;

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
                ret.stopWords = stopWords;
                ret.setCache(vocabCache);
                ret.minWordFrequency = minWordFrequency;
                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }
                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }

            else {
                Word2Vec ret = new Word2Vec(iter);
                ret.layerSize = layerSize;
                ret.window = window;
                ret.stopWords = stopWords;
                ret.minWordFrequency = minWordFrequency;
                ret.setCache(vocabCache);

                ret.minWordFrequency = minWordFrequency;


                try {
                    if (tokenizerFactory == null)
                        tokenizerFactory = new UimaTokenizerFactory();
                }catch(Exception e) {
                    throw new RuntimeException(e);
                }


                ret.tokenizerFactory = tokenizerFactory;
                return ret;
            }


        }
    }




}
