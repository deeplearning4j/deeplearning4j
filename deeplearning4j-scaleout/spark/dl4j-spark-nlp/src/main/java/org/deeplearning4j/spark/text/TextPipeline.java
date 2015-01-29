package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.stopwords.StopWords;

import java.util.List;

/**
 * A spark based text pipeline
 *
 * @author Adam Gibson
 */
public class TextPipeline {
    private JavaRDD<String> corpus;
    private List<String> stopWords;
    private int minWordFrequency = 5;

    public TextPipeline(JavaRDD<String> corpus, List<String> stopWords, int minWordFrequency) {
        this.corpus = corpus;
        this.stopWords = stopWords;
        this.minWordFrequency = minWordFrequency;
    }

    public TextPipeline(JavaRDD<String> corpus) {
        this(corpus, StopWords.getStopWords(),5);
    }

    public TextPipeline(JavaRDD<String> corpus, int minWordFrequency) {
        this(corpus,StopWords.getStopWords(),minWordFrequency);
    }

    public Pair<VocabCache,Long> process() {
        JavaSparkContext sc = new JavaSparkContext(corpus.context());
        Broadcast<List<String>> broadcast = sc.broadcast(stopWords);
        return corpus.map(new TokenizerFunction()).map(new VocabCacheFunction(minWordFrequency,broadcast)).cache().collect().get(0);
    }


}
