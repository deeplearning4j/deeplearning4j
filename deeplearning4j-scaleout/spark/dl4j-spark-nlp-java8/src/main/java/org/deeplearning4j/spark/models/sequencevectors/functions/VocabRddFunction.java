package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.util.ArrayList;
import java.util.List;

/**
 * This function builds RDD of vocab words
 *
 * On top of that, we use this Function to launch VoidParameterServer
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class VocabRddFunction<T extends SequenceElement> implements FlatMapFunction<Sequence<T>, T> {
    protected Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast;
    protected Broadcast<VoidConfiguration> paramServerConfigurationBroadcast;

    protected transient VectorsConfiguration configuration;

    public VocabRddFunction(@NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast, @NonNull Broadcast<VoidConfiguration> paramServerConfigurationBroadcast) {
        this.vectorsConfigurationBroadcast = vectorsConfigurationBroadcast;
        this.paramServerConfigurationBroadcast = paramServerConfigurationBroadcast;

        log.info("VocabRDDFunction constructor");
    }

    @Override
    public Iterable<T> call(Sequence<T> sequence) throws Exception {
        if (configuration == null)
            configuration = vectorsConfigurationBroadcast.getValue();

        log.info("Initializing VoidParameterServer...");
        System.out.println("Initializing VPS...");

        // we just silently initialize server
        VoidParameterServer.getInstance().init(paramServerConfigurationBroadcast.getValue());

        // TODO: call for initializeSeqVec here

        List<T> elements = new ArrayList<>();

        elements.addAll(sequence.getElements());

        // FIXME: this is PROBABLY bad, we might want to ensure, there's no duplicates.
        if (configuration.isTrainSequenceVectors())
            if (sequence.getSequenceLabels().size() > 0)
                elements.addAll(sequence.getSequenceLabels());

        return elements;
    }
}
