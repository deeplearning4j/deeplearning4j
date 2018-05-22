package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkElementsLearningAlgorithm;
import org.deeplearning4j.spark.models.sequencevectors.learning.SparkSequenceLearningAlgorithm;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This is wrapper for SequenceVectors training over given Sequence<T>
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TrainingFunction<T extends SequenceElement> implements VoidFunction<Sequence<T>> {
    protected Broadcast<VocabCache<ShallowSequenceElement>> vocabCacheBroadcast;
    protected Broadcast<VectorsConfiguration> configurationBroadcast;
    protected Broadcast<VoidConfiguration> paramServerConfigurationBroadcast;

    protected transient VoidParameterServer paramServer;
    protected transient VectorsConfiguration vectorsConfiguration;

    protected transient SparkElementsLearningAlgorithm elementsLearningAlgorithm;
    protected transient SparkSequenceLearningAlgorithm sequenceLearningAlgorithm;
    protected transient VocabCache<ShallowSequenceElement> shallowVocabCache;

    protected transient TrainingDriver<? extends TrainingMessage> driver;

    public TrainingFunction(@NonNull Broadcast<VocabCache<ShallowSequenceElement>> vocabCacheBroadcast,
                    @NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast,
                    @NonNull Broadcast<VoidConfiguration> paramServerConfigurationBroadcast) {
        this.vocabCacheBroadcast = vocabCacheBroadcast;
        this.configurationBroadcast = vectorsConfigurationBroadcast;
        this.paramServerConfigurationBroadcast = paramServerConfigurationBroadcast;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void call(Sequence<T> sequence) throws Exception {
        /**
         * Depending on actual training mode, we'll either go for SkipGram/CBOW/PV-DM/PV-DBOW or whatever
         */
        if (vectorsConfiguration == null)
            vectorsConfiguration = configurationBroadcast.getValue();

        if (paramServer == null) {
            paramServer = VoidParameterServer.getInstance();

            if (elementsLearningAlgorithm == null) {
                try {
                    elementsLearningAlgorithm = (SparkElementsLearningAlgorithm) Class
                                    .forName(vectorsConfiguration.getElementsLearningAlgorithm()).newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            driver = elementsLearningAlgorithm.getTrainingDriver();

            // FIXME: init line should probably be removed, basically init happens in VocabRddFunction
            paramServer.init(paramServerConfigurationBroadcast.getValue(), new RoutedTransport(), driver);
        }

        if (vectorsConfiguration == null)
            vectorsConfiguration = configurationBroadcast.getValue();

        if (shallowVocabCache == null)
            shallowVocabCache = vocabCacheBroadcast.getValue();


        if (elementsLearningAlgorithm == null && vectorsConfiguration.getElementsLearningAlgorithm() != null) {
            // TODO: do ELA initialization
            try {
                elementsLearningAlgorithm = (SparkElementsLearningAlgorithm) Class
                                .forName(vectorsConfiguration.getElementsLearningAlgorithm()).newInstance();
                elementsLearningAlgorithm.configure(shallowVocabCache, null, vectorsConfiguration);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (sequenceLearningAlgorithm == null && vectorsConfiguration.getSequenceLearningAlgorithm() != null) {
            // TODO: do SLA initialization
            try {
                sequenceLearningAlgorithm = (SparkSequenceLearningAlgorithm) Class
                                .forName(vectorsConfiguration.getSequenceLearningAlgorithm()).newInstance();
                sequenceLearningAlgorithm.configure(shallowVocabCache, null, vectorsConfiguration);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (elementsLearningAlgorithm == null && sequenceLearningAlgorithm == null) {
            throw new ND4JIllegalStateException("No LearningAlgorithms specified!");
        }


        /*
         at this moment we should have everything ready for actual initialization
         the only limitation we have - our sequence is detached from actual vocabulary, so we need to merge it back virtually
        */
        Sequence<ShallowSequenceElement> mergedSequence = new Sequence<>();
        for (T element : sequence.getElements()) {
            // it's possible to get null here, i.e. if frequency for this element is below minWordFrequency threshold
            ShallowSequenceElement reduced = shallowVocabCache.tokenFor(element.getStorageId());

            if (reduced != null)
                mergedSequence.addElement(reduced);
        }

        // do the same with labels, transfer them, if any
        if (sequenceLearningAlgorithm != null && vectorsConfiguration.isTrainSequenceVectors()) {
            for (T label : sequence.getSequenceLabels()) {
                ShallowSequenceElement reduced = shallowVocabCache.tokenFor(label.getStorageId());

                if (reduced != null)
                    mergedSequence.addSequenceLabel(reduced);
            }
        }


        // now we have shallow sequence, which we'll use for training
        /**
         * All we want here, is uniform way to do training, that's matching both standalone and spark codebase.
         * So we need some neat method, that takes sequence as input, and returns **something** that's either used for aggregation, or for ParamServer message
         */
        // FIXME: temporary hook
        if (sequence.size() > 0)
            paramServer.execDistributed(
                            elementsLearningAlgorithm.frameSequence(mergedSequence, new AtomicLong(119), 25e-3));
        else
            log.warn("Skipping empty sequence...");

    }
}
