package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.stopwords.StopWords;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

import java.util.List;

/**
 * A spark based text pipeline
 * with minimum word frequency and stop words
 *
 * @author Adam Gibson
 */
public class TextPipeline {
    private JavaRDD<String> corpus;
    private List<String> stopWords;
    private int minWordFrequency = 5;



    /**
     *
     * @param corpus the corpus of text
     * @param stopWords the stop words to use
     * @param minWordFrequency the minimum word frequency for the vocab
     */
    public TextPipeline(JavaRDD<String> corpus, List<String> stopWords, int minWordFrequency) {
        this.corpus = corpus;
        this.stopWords = stopWords;
        this.minWordFrequency = minWordFrequency;
    }

    /**
     * Create a text pipeline with the given corpus,
     * StopWords.getStopWords() and a minimum word frequency of 5
     * @param corpus the corpus to use
     */
    public TextPipeline(JavaRDD<String> corpus) {
        this(corpus, StopWords.getStopWords(),5);
    }

    /**
     * Create a text pipeline with the specified corpus
     * @param corpus the corpus to use
     * @param minWordFrequency the minimum word frequency to use
     */
    public TextPipeline(JavaRDD<String> corpus, int minWordFrequency) {
        this(corpus,StopWords.getStopWords(),minWordFrequency);
    }

    /**
     * Get a vocab cache with all of the vocab based on the
     * specified stop words and minimum word frequency
     * @param tokenizer the fully qualified class name to use for instantiating tokenizers
     * @return the vocab cache and associated total number of words
     */
    public Pair<VocabCache,Long> process(String tokenizer) {
        JavaSparkContext sc = new JavaSparkContext(corpus.context());
        Broadcast<List<String>> broadcast = sc.broadcast(stopWords);
        return corpus.map(new TokenizerFunction(tokenizer))
                .map(new VocabCacheFunction(minWordFrequency,new InMemoryLookupCache(),broadcast))
                .cache().collect().get(0);
    }

    /**
     * Get a vocab cache with all of the vocab based on the
     * specified stop words and minimum word frequency
     * @return the vocab cache and associated total number of words
     */
    public Pair<VocabCache,Long> process() {
        JavaSparkContext sc = new JavaSparkContext(corpus.context());
        Broadcast<List<String>> broadcast = sc.broadcast(stopWords);
        return corpus.map(new TokenizerFunction(DefaultTokenizerFactory.class.getName()))
                .map(new VocabCacheFunction(minWordFrequency,new InMemoryLookupCache(),broadcast))
                .cache().collect().get(0);
    }


}
