package org.deeplearning4j.spark.models.embeddings.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * This is wrapper for SequenceVectors training over given Sequence<T>
 *
 * @author raver119@gmail.com
 */
public class TrainingFunction<T extends SequenceElement> implements VoidFunction<Sequence<T>> {
    protected Broadcast<VocabCache<T>> vocabCacheBroadcast;
    protected Broadcast<VectorsConfiguration> configurationBroadcast;

    public TrainingFunction(@NonNull Broadcast<VocabCache<T>> vocabCacheBroadcast, @NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast) {
        this.vocabCacheBroadcast = vocabCacheBroadcast;
        this.configurationBroadcast = vectorsConfigurationBroadcast;
    }

    @Override
    public void call(Sequence<T> sequence) throws Exception {
        /**
         * Depending on actual training mode, we'll either go for SkipGram/CBOW/PV-DM/PV-DBOW or whatever
         */
    }
}
