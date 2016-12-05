package org.deeplearning4j.spark.models.embeddings.sequencevectors;

import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.rdd.RDD;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.functions.CountFunction;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.functions.ElementsFrequenciesAccumulator;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.functions.ListSequenceConvertFunction;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.functions.TrainingFunction;

import java.util.List;

/**
 * Generic SkipGram/CBOW implementation for dl4j-spark-nlp
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkSequenceVectors<T extends SequenceElement> extends SequenceVectors<T> {
    protected Accumulator<Counter<Long>> elementsFreqAccum;
    protected StorageLevel storageLevel = StorageLevel.MEMORY_ONLY();


    protected Broadcast<VocabCache<T>> vocabCacheBroadcast;
    protected Broadcast<VectorsConfiguration> configurationBroadcast;

    protected SparkSequenceVectors() {

    }

    /**
     * PLEASE NOTE: This method isn't supported for Spark implementation. Consider using fitLists() or fitSequences() instead.
     */
    @Override
    public void fit() {
        throw new UnsupportedOperationException("To use fit() method, please consider using standalone implementation");
    }

    /**
     * Utility method. fitSequences() used within.
     *
     * PLEASE NOTE: This method can't be used to train for labels, since List<T> can't hold labels. If you need labels - consider manual Sequence creation instead.
     *
     * @param corpus
     */
    public void fitLists(JavaRDD<List<T>> corpus) {
        // we just convert List to sequences
        JavaRDD<Sequence<T>> rdd = corpus.map(new ListSequenceConvertFunction<T>());

        // and use fitSequences()
        fitSequences(rdd);
    }

    /**
     * Base training entry point
     *
     * @param corpus
     */
    public void fitSequences(JavaRDD<Sequence<T>> corpus) {
        /**
         * Basically all we want for base implementation here is 3 things:
         * a) build vocabulary
         * b) build huffman tree
         * c) do training
         *
         * in this case all classes extending SeqVec, like deepwalk or word2vec will be just building their RDD<Sequence<T>>,
         * and calling this method for training, instead implementing own routines
         */

        if (workers > 1)
            corpus.repartition(workers);

        if (storageLevel != null)
            corpus.persist(storageLevel);

        final JavaSparkContext sc = new JavaSparkContext(corpus.context());

        // set up freqs accumulator
        elementsFreqAccum = corpus.context().accumulator(new Counter<Long>(), new ElementsFrequenciesAccumulator());
        CountFunction<T> elementsCounter = new CountFunction<>(elementsFreqAccum, false);

        // count all sequence elements and their sum
        JavaRDD<Pair<Sequence<T>, Long>> countedCorpus = corpus.map(elementsCounter);

        // just to trigger map function, since we need huffman tree before proceeding
        long numberOfSequences = countedCorpus.count();

        // now we grab counter, which contains frequencies for all SequenceElements in corpus
        Counter<Long> finalCounter = elementsFreqAccum.value();

        long numberOfElements = (long) finalCounter.totalCount();

        long numberOfUniqueElements = finalCounter.size();

        log.info("Total number of sequences: {}; Total number of elements entries: {}; Total number of unique elements: {}", numberOfSequences, numberOfElements, numberOfUniqueElements);

        /*
         build RDD of reduced SequenceElements, just get rid of labels temporary, stick to some numerical values,
         like index or hashcode. So we could reduce driver memory footprint
         */


        // build huffman tree, and update original RDD with huffman encoding info
        VocabCache<T> vocabCache = buildVocabularyFromCounter(finalCounter);
        // TODO: right at this place we should launch one more map, that will update original RDD with huffman encoding
        vocabCacheBroadcast = sc.broadcast(vocabCache);
        configurationBroadcast = sc.broadcast(configuration);


        // proceed to training
        TrainingFunction<T> trainer = new TrainingFunction<>(vocabCacheBroadcast, configurationBroadcast);

        if (configuration != null)
            for (int e = 0; e < configuration.getEpochs(); e++)
                corpus.foreach(trainer);


        // at this particular moment training should be pretty much done, and we're good to go for export


        // unpersist, if we've persisten corpus after all
        if (storageLevel != null)
            corpus.unpersist();
    }

    /**
     * This method updates builds limited RDD of Sequence<T> with Huffman info embedded
     *
     * @param counter
     * @return
     */
    protected VocabCache<T> buildVocabularyFromCounter(Counter<Long> counter) {

        // TODO: need simplified cache here, that will operate on Long instead of string labels
        VocabCache<T> vocabCache = new AbstractCache<>();
        for (Long id : counter.keySet()) {
            // TODO: to be implemented
        }

        // TODO: build huffman tree here

        return vocabCache;
    }

    protected Counter<Long> getCounter() {
        return elementsFreqAccum.value();
    }
}
