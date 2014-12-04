package org.deeplearning4j.models.glove;

import org.deeplearning4j.bagofwords.vectorizer.TextVectorizer;
import org.deeplearning4j.bagofwords.vectorizer.TfidfVectorizer;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.parallel.Parallelization;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.SetUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Glove by socher et. al
 *
 * @author Adam Gibson
 */
public class Glove implements Serializable {

    private VocabCache cache;
    private SentenceIterator sentenceIterator;
    private TextVectorizer textVectorizer;
    private TokenizerFactory tokenizerFactory;
    private GloveWeightLookupTable lookupTable;
    private int layerSize = 100;
    private double learningRate = 0.1;
    private double xMax = 0.75;
    private int windowSize = 15;
    private CoOccurrences coOccurrences;
    private List<String> stopWords = StopWords.getStopWords();
    private boolean stem = false;
    protected Queue<List<List<VocabWord>>> jobQueue = new LinkedBlockingDeque<>(10000);
    private int batchSize = 1000;
    private int minWordFrequency = 5;
    public final static String UNK = Word2Vec.UNK;

    public Glove(VocabCache cache, SentenceIterator sentenceIterator, TextVectorizer textVectorizer, TokenizerFactory tokenizerFactory, GloveWeightLookupTable lookupTable, int layerSize, double learningRate, double xMax, int windowSize, CoOccurrences coOccurrences, List<String> stopWords, boolean stem,int batchSize,int minWordFrequency) {
        this.cache = cache;
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
    }

    public void fit() {
        if(cache == null)
            cache = new InMemoryLookupCache();

        if(textVectorizer == null) {
            textVectorizer = new TfidfVectorizer.Builder().tokenize(tokenizerFactory)
                    .cache(cache).iterate(sentenceIterator).minWords(minWordFrequency)
                    .stopWords(stopWords).stem(stem).build();

            textVectorizer.fit();
        }


        sentenceIterator.reset();

        if(coOccurrences == null) {
            coOccurrences = new CoOccurrences.Builder()
                    .cache(cache).iterate(sentenceIterator)
                    .tokenizer(tokenizerFactory).windowSize(windowSize)
                    .build();

            coOccurrences.fit();

        }

        if(lookupTable == null)
            lookupTable = new GloveWeightLookupTable.Builder().xMax(xMax)
                    .cache(cache).lr(learningRate).vectorLength(layerSize)
                    .build();


        if(lookupTable.getSyn0() == null)
            lookupTable.resetWeights();

        final List<List<VocabWord>> miniBatches = new CopyOnWriteArrayList<>();

        for(String s : coOccurrences.getCoOCurreneCounts().keySet()) {
            for(String s1 : coOccurrences.getCoOCurreneCounts().getCounter(s).keySet()) {
                VocabWord vocabWord = cache.wordFor(s);
                VocabWord vocabWord1 = cache.wordFor(s1);
                miniBatches.add(Arrays.asList(vocabWord, vocabWord1));
                if(miniBatches.size() >= batchSize) {
                    jobQueue.add(new ArrayList<>(miniBatches));
                    miniBatches.clear();
                }
            }
        }




        if(!miniBatches.isEmpty())
            jobQueue.add(miniBatches);


        final AtomicInteger processed = new AtomicInteger(coOccurrences.getCoOCurreneCounts().size());


        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        Parallelization.runInParallel(Runtime.getRuntime().availableProcessors(),new Runnable() {
            @Override
            public void run() {
                while(processed.get() > 0) {
                    List<List<VocabWord>> batch = jobQueue.poll();
                    if(batch == null)
                        continue;

                    for(List<VocabWord> list : batch) {
                        VocabWord w1 = list.get(0);
                        VocabWord w2 = list.get(1);
                        double weight = coOccurrences.getCoOCurreneCounts().getCount(w1.getWord(),w2.getWord());
                        lookupTable.iterateSample(w1, w2, learningRate, weight);
                        processed.decrementAndGet();

                    }

                }
            }
        });




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
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;
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
        INDArray words = Nd4j.create(positive.size() + negative.size(),layerSize);
        int row = 0;
        Set<String> union = SetUtils.union(new HashSet<>(positive),new HashSet<>(negative));
        for(String s : positive) {
            words.putRow(row++,lookupTable.vector(s));
        }

        for(String s : negative) {
            words.putRow(row++,lookupTable.vector(s).mul(-1));
        }

        INDArray mean = words.mean(0);
        if(lookupTable instanceof  InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;
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


        if(lookupTable instanceof  InMemoryLookupTable) {
            InMemoryLookupTable l = (InMemoryLookupTable) lookupTable;
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
        private double learningRate = 0.5;
        private double xMax = 0.75;
        private int windowSize = 5;
        private CoOccurrences coOccurrences;
        private List<String> stopWords = StopWords.getStopWords();
        private boolean stem = false;
        private int batchSize = 100;
        private int minWordFrequency = 5;

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
            return new Glove(vocabCache, sentenceIterator, textVectorizer, tokenizerFactory, weightLookupTable, layerSize, learningRate, xMax, windowSize, coOccurrences, stopWords, stem, batchSize,minWordFrequency);
        }
    }

}
