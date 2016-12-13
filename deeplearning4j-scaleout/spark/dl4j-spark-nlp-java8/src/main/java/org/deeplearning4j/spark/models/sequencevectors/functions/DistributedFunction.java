package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.nd4j.parameterserver.distributed.conf.Configuration;

/**
 *
 *
 * @author raver119@gmail.coms
 */
public class DistributedFunction<T extends SequenceElement> implements Function<T, ExportContainer<T>> {

    protected Broadcast<Configuration> configurationBroadcast;
    protected Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast;
    protected Broadcast<VocabCache<ShallowSequenceElement>> shallowVocabBroadcast;

    protected transient VocabCache<ShallowSequenceElement> shallowVocabCache;

    public DistributedFunction(@NonNull Broadcast<Configuration> configurationBroadcast, @NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast, @NonNull Broadcast<VocabCache<ShallowSequenceElement>> shallowVocabBroadcast) {
        this.configurationBroadcast = configurationBroadcast;
        this.vectorsConfigurationBroadcast = vectorsConfigurationBroadcast;
        this.shallowVocabBroadcast = shallowVocabBroadcast;
    }

    @Override
    public ExportContainer<T> call(T word) throws Exception {
        if (shallowVocabCache == null)
            shallowVocabCache = shallowVocabBroadcast.getValue();

        ExportContainer<T> container = new ExportContainer<>();

        container.setElement(word);

        // TODO: request data from VoidParameterServer

        return container;
    }
}
