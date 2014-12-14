package org.deeplearning4j.models.glove;

import com.google.common.collect.Lists;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.SetUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Glove by socher et. al
 *
 * @author Adam Gibson
 */
public class Glove implements Serializable {

    private VocabCache cache;
    private transient SentenceIterator sentenceIterator;
    private transient TextVectorizer textVectorizer;
    private transient TokenizerFactory tokenizerFactory;
    private GloveWeightLookupTable lookupTable;
    private int layerSize = 100;
    private double learningRate = 0.05;
    private double xMax = 0.75;
    private int windowSize = 15;
    private CoOccurrences coOccurrences;
    private List<String> stopWords = StopWords.getStopWords();
    private boolean stem = false;
    protected Queue<Pair<Integer,List<Pair<VocabWord,VocabWord>>>> jobQueue = new LinkedBlockingDeque<>();
    private int batchSize = 1000;
    private int minWordFrequency = 5;
    private double maxCount = 100;
    public final static String UNK = Word2Vec.UNK;
    private int iterations = 5;
    private static Logger log = LoggerFactory.getLogger(Glove.class);
    private boolean symmetric = true;
    private transient RandomGenerator gen;
    private boolean shuffle = true;
    private transient Random shuffleRandom;
    private int numWorkers = Runtime.getRuntime().availableProcessors();

    private Glove(){}

    public Glove(VocabCache cache, SentenceIterator sentenceIterator, TextVectorizer textVectorizer, TokenizerFactory tokenizerFactory, GloveWeightLookupTable lookupTable, int layerSize, double learningRate, double xMax, int windowSize, CoOccurrences coOccurrences, List<String> stopWords, boolean stem,int batchSize,int minWordFrequency,double maxCount,int iterations,boolean symmetric,RandomGenerator gen,boolean shuffle,long seed,int numWorkers) {
        this.numWorkers = numWorkers;
        this.gen = gen;
        this.cache = cache;
        this.shuffle = shuffle;
        this.sentenceIterator = sentenceIterator;
        this.textVectorizer = textVectorizer;
        this.tokenizerFactory = tokenizerFactory;
        this.lookupTable = lookupTable;
        this.layerSize = layerSize;
        this.learningRate = learningRate;
        this.xMax = xMax;
        this.windowSize = windowSize;
        this.coOccurrences = coOccurrences;
        this.stopWords = stopWords;
        this.stem = stem;
        this.batchSize = batchSize;
        this.minWordFrequency = minWordFrequency;
        this.maxCount = maxCount;
        this.iterations = iterations;
        this.symmetric = symmetric;
        shuffleRandom = new Random(seed);
    }

    public void fit() {
        boolean cacheFresh = false;

        if(cache == null) {
            cacheFresh  = true;
            cache = new InMemoryLookupCache();
        }

        if(textVectorizer == null && cacheFresh) {
            textVectorizer = new TfidfVectorizer.Builder().tokenize(tokenizerFactory)
                    .cache(cache).iterate(sentenceIterator).minWords(minWordFrequency)
                    .stopWords(stopWords).stem(stem).build();

            textVectorizer.fit();
        }

        if(sentenceIterator != null)
            sentenceIterator.reset();

        if(coOccurrences == null) {
            coOccurrences = new CoOccurrences.Builder()
                    .cache(cache).iterate(sentenceIterator).symmetric(symmetric)
                    .tokenizer(tokenizerFactory).windowSize(windowSize)
                    .build();

            coOccurrences.fit();

        }

        if(lookupTable == null)
            lookupTable = new GloveWeightLookupTable.Builder().xMax(xMax).maxCount(maxCount)
                    .cache(cache).lr(learningRate).vectorLength(layerSize).gen(gen)
                    .build();


        if(lookupTable.getSyn0() == null)
            lookupTable.resetWeights();
        final List<Pair<String,String>> pairList = coOccurrences.coOccurrenceList();
        if(shuffle)
            Collections.shuffle(pairList,shuffleRandom);



        final AtomicInteger countUp = new AtomicInteger(0);
        final Counter<Integer> erroriPerIteration = Util.parallelCounter();
        log.info("Processing # of co occurrences " + coOccurrences.numCoOccurrences());
        for(int i = 0; i < iterations; i++) {
            final AtomicInteger processed = new AtomicInteger(coOccurrences.numCoOccurrences());
            doIteration(i, pairList, erroriPerIteration, processed, countUp);
            log.info("Processed " + countUp.doubleValue() + " out of " + (pairList.size() * iterations) + " error was " + erroriPerIteration.getCount(i));

        }


    }


    public void doIteration(int i,List<Pair<String,String>> pairList, final Counter<Integer> errorPerIteration,final AtomicInteger processed,final AtomicInteger countUp) {
        log.info("Iteration " + i);
        if(shuffle)
            Collections.shuffle(pairList,shuffleRandom);
        List<List<Pair<String,String>>> miniBatches = Lists.partition(pairList,batchSize);
        int count = 0;
        for(List<Pair<String,String>> batch : miniBatches) {
            List<Pair<VocabWord,VocabWord>> send = new ArrayList<>();
            for (Pair<String, String> next : batch) {
                String w1 = next.getFirst();
                String w2 = next.getSecond();
                VocabWord vocabWord = cache.wordFor(w1);
                VocabWord vocabWord1 = cache.wordFor(w2);
                send.add(new Pair<>(vocabWord, vocabWord1));

            }

            jobQueue.add(new Pair<>(i,send));
            log.info("Queued batch " + count + " of " + miniBatches.size());
            count++;
        }


        Parallelization.runInParallel(numWorkers,new Runnable() {
            @Override
            public void run() {
                while(processed.get() > 0 || !jobQueue.isEmpty()) {
                    Pair<Integer,List<Pair<VocabWord,VocabWord>>> work = jobQueue.poll();
                    if(work == null)
                        continue;
                    List<Pair<VocabWord,VocabWord>> batch = work.getSecond();

                    for(Pair<VocabWord,VocabWord> pair : batch) {
                        VocabWord w1 = pair.getFirst();
                        VocabWord w2 = pair.getSecond();
                        double weight = getCount(w1.getWord(),w2.getWord());
                        if(weight <= 0) {
                            countUp.incrementAndGet();
                            processed.decrementAndGet();
                            continue;

                        }
                        errorPerIteration.incrementCount(work.getFirst(),lookupTable.iterateSample(w1,w2,weight));
                        countUp.incrementAndGet();
                        if(countUp.get() % 10000 == 0)
                            log.info("Processed " + countUp.get() + " co occurrences");
                        processed.decrementAndGet();
                    }




                }
            }
        },true);
    }


    /**
     * Load a glove model from an input stream.
     * The format is:
     * word num1 num2....
     * @param is the input stream to read from for the weights
     * @param biases the bias input stream
     * @return the loaded model
     * @throws IOException if one occurs
     */
    public static Glove load(InputStream is,InputStream biases) throws IOException {
        LineIterator iter = IOUtils.lineIterator(is,"UTF-8");
        Glove glove = new Glove();
        Map<String,float[]> wordVectors = new HashMap<>();
        int count = 0;
        while(iter.hasNext()) {
            String line = iter.nextLine().trim();
            if(line.isEmpty())
                continue;
            String[] split = line.split(" ");
            String word = split[0];
            if(glove.cache == null)
                glove.cache = new InMemoryLookupCache();

            if(glove.getLookupTable() == null) {
                glove.lookupTable = new GloveWeightLookupTable.Builder()
                        .cache(glove.cache).vectorLength(split.length - 1)
                        .build();

            }

            if(word.isEmpty())
                continue;
            float[] read = read(split,glove.lookupTable.getVectorLength());
            if(read.length < 1)
                continue;

            VocabWord w1 = new VocabWord(1,word);
            glove.layerSize = glove.lookupTable.getVectorLength();
            w1.setIndex(count);
            glove.cache.addToken(w1);
            glove.cache.addWordToIndex(count, word);
            glove.cache.putVocabWord(word);
            wordVectors.put(word,read);
            count++;



        }

        glove.lookupTable.setSyn0(weights(glove,wordVectors));



        iter.close();

        glove.lookupTable.setBias(Nd4j.readTxt(biases," "));

        return glove;

    }




    private static INDArray weights(Glove glove,Map<String,float[]> data) {
        INDArray ret = Nd4j.create(data.size(),glove.getLookupTable().getVectorLength());
        for(String key : data.keySet()) {
            INDArray row = Nd4j.create(Nd4j.createBuffer(data.get(key)));
            if(row.length() != glove.getLookupTable().getVectorLength())
                continue;
            if(glove.getCache().indexOf(key) >= data.size())
                continue;
            ret.putRow(glove.getCache().indexOf(key), row);
        }
        return ret;
    }


    private static float[] read(String[] split,int length) {
        float[] ret = new float[length];
        for(int i = 1; i < split.length; i++) {
            ret[i - 1] = Float.parseFloat(split[i]);
        }
        return ret;
    }


    public double getCount(String w1,String w2) {
        return coOccurrences.getCoOCurreneCounts().getCount(w1,w2);
    }

    public CoOccurrences getCoOccurrences() {
        return coOccurrences;
    }

    public void setCoOccurrences(CoOccurrences coOccurrences) {
        this.coOccurrences = coOccurrences;
    }

    /**
     * Accuracy based on questions which are a space separated list of strings
     * where the first word is the query word, the next 2 words are negative,
     * and the last word is the predicted word to be nearest
     * @param questions the questions to ask
     * @return the accuracy based on these questions
     */
    public Map<String,Double> accuracy(List<String> questions) {
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
            if(MathUtils.stringSimilarity(word, s) >= accuracy)
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
            return lookupTable.vector(UNK).ravel().data().asDouble();
        return lookupTable.vector(word).ravel().data().asDouble();
    }

    /**
     * Get the word vector for a given matrix
     * @param word the word to get the matrix for
     * @return the ndarray for this word
     */
    public INDArray getWordVectorMatrix(String word) {
        int i = this.cache.indexOf(word);
        if(i < 0)
            return lookupTable.vector(UNK);
        return lookupTable.vector(word);
    }

    /**
     * Returns the word vector divided by the norm2 of the array
     * @param word the word to get the matrix for
     * @return the looked up matrix
     */
    public INDArray getWordVectorMatrixNormalized(String word) {
        int i = this.cache.indexOf(word);

        if(i < 0)
            return lookupTable.vector(UNK);
        INDArray r =  lookupTable.vector(word);
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
        Set<String> union = SetUtils.union(new HashSet<>(positive), new HashSet<>(negative));
        for(String s : positive)
            words.addi(lookupTable.vector(s));


        for(String s : negative)
            words.addi(lookupTable.vector(s).mul(-1));


        if(lookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l =  lookupTable;
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
            double sim = Transforms.cosineSim(words, otherVec);
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


        if(lookupTable instanceof InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) cache;
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
        for(String p : SetUtils.union(new HashSet<>(positive),new HashSet<>(negative))) {
            if(!cache.containsWord(p)) {
                return new ArrayList<>();
            }
        }


        INDArray words = Nd4j.create(positive.size() + negative.size(), layerSize);
        int row = 0;
        Set<String> union = SetUtils.union(new HashSet<>(positive),new HashSet<>(negative));
        for(String s : positive) {
            words.putRow(row++,lookupTable.vector(s));
        }

        for(String s : negative) {
            words.putRow(row++, lookupTable.vector(s).mul(-1));
        }

        INDArray mean = words.isMatrix() ? words.mean(0) : words;
        if(lookupTable instanceof  InMemoryLookupTable) {
            InMemoryLookupTable l =  lookupTable;
            INDArray syn0 = l.getSyn0();
            INDArray weights = mean;
            INDArray distances = syn0.mmul(weights.transpose());
            distances.diviRowVector(distances.norm2(1));
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
        return wordsNearest(Arrays.asList(word),new ArrayList<String>(),n);

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
        return  Nd4j.getBlasWrapper().dot(vector, vector2);
    }


    public VocabCache getCache() {
        return cache;
    }

    public void setCache(VocabCache cache) {
        this.cache = cache;
    }

    public GloveWeightLookupTable getLookupTable() {
        return lookupTable;
    }

    public void setLookupTable(GloveWeightLookupTable lookupTable) {
        this.lookupTable = lookupTable;
    }

    public static class Builder {
        private VocabCache vocabCache;
        private SentenceIterator sentenceIterator;
        private TextVectorizer textVectorizer;
        private TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        private GloveWeightLookupTable weightLookupTable;
        private int layerSize = 300;
        private double learningRate = 0.05;
        private double xMax = 0.75;
        private int windowSize = 5;
        private CoOccurrences coOccurrences;
        private List<String> stopWords = StopWords.getStopWords();
        private boolean stem = false;
        private int batchSize = 100;
        private int minWordFrequency = 5;
        private double maxCount = 100;
        private int iterations = 5;
        private boolean symmetric = true;
        private boolean shuffle = true;
        private long seed = 123;
        private int numWorkers = Runtime.getRuntime().availableProcessors();
        private RandomGenerator gen = new MersenneTwister(seed);


        public Builder numWorkers(int numWorkers) {
            this.numWorkers = numWorkers;
            return this;
        }

        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        public Builder shuffle(boolean shuffle) {
            this.shuffle = shuffle;
            return this;
        }
        public Builder rng(RandomGenerator gen) {
            this.gen = gen;
            return this;
        }

        public Builder symmetric(boolean symmetric) {
            this.symmetric = symmetric;
            return this;
        }

        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }

        public Builder maxCount(double maxCount) {
            this.maxCount = maxCount;
            return this;
        }

        public Builder minWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder cache(VocabCache vocabCache) {
            this.vocabCache = vocabCache;
            return this;
        }

        public Builder iterate(SentenceIterator sentenceIterator) {
            this.sentenceIterator = sentenceIterator;
            return this;
        }

        public Builder vectorizer(TextVectorizer textVectorizer) {
            this.textVectorizer = textVectorizer;
            return this;
        }

        public Builder tokenizer(TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder weights(GloveWeightLookupTable weightLookupTable) {
            this.weightLookupTable = weightLookupTable;
            return this;
        }

        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public Builder learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder xMax(double xMax) {
            this.xMax = xMax;
            return this;
        }

        public Builder windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder coOccurrences(CoOccurrences coOccurrences) {
            this.coOccurrences = coOccurrences;
            return this;
        }

        public Builder stopWords(List<String> stopWords) {
            this.stopWords = stopWords;
            return this;
        }

        public Builder stem(boolean stem) {
            this.stem = stem;
            return this;
        }

        public Builder batchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }

        public Glove build() {
            return new Glove(vocabCache, sentenceIterator, textVectorizer, tokenizerFactory, weightLookupTable, layerSize, learningRate, xMax, windowSize, coOccurrences, stopWords, stem, batchSize,minWordFrequency,maxCount,iterations,symmetric,gen,shuffle,seed,numWorkers);
        }
    }

}
