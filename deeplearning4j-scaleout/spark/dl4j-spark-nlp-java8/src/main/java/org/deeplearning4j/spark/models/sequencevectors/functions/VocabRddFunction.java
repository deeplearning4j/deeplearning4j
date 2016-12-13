package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class VocabRddFunction<T extends SequenceElement> implements FlatMapFunction<Sequence<T>, T> {
    protected Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast;

    protected transient VectorsConfiguration configuration;

    public VocabRddFunction(@NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast) {
        this.vectorsConfigurationBroadcast = vectorsConfigurationBroadcast;
    }

    @Override
    public Iterable<T> call(Sequence<T> sequence) throws Exception {
        if (configuration == null)
            configuration = vectorsConfigurationBroadcast.getValue();

        List<T> elements = new ArrayList<>();

        elements.addAll(sequence.getElements());

        // FIXME: this is PROBABLY bad, we might want to ensure, there's no duplicates.
        if (configuration.isTrainSequenceVectors())
            if (sequence.getSequenceLabels().size() > 0)
                elements.addAll(sequence.getSequenceLabels());

        return elements;
    }
}
