package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
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
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class PartitionTrainingFunction<T extends SequenceElement> implements VoidFunction<Iterator<Sequence<T>>> {
    protected Broadcast<VocabCache<ShallowSequenceElement>> vocabCacheBroadcast;
    protected Broadcast<VectorsConfiguration> configurationBroadcast;
    protected Broadcast<VoidConfiguration> paramServerConfigurationBroadcast;

    protected transient VoidParameterServer paramServer;
    protected transient VectorsConfiguration vectorsConfiguration;

    protected transient SparkElementsLearningAlgorithm elementsLearningAlgorithm;
    protected transient SparkSequenceLearningAlgorithm sequenceLearningAlgorithm;
    protected transient VocabCache<ShallowSequenceElement> shallowVocabCache;

    protected transient TrainingDriver<? extends TrainingMessage> driver;

    public PartitionTrainingFunction(@NonNull Broadcast<VocabCache<ShallowSequenceElement>> vocabCacheBroadcast,
                    @NonNull Broadcast<VectorsConfiguration> vectorsConfigurationBroadcast,
                    @NonNull Broadcast<VoidConfiguration> paramServerConfigurationBroadcast) {
        this.vocabCacheBroadcast = vocabCacheBroadcast;
        this.configurationBroadcast = vectorsConfigurationBroadcast;
        this.paramServerConfigurationBroadcast = paramServerConfigurationBroadcast;
    }

    @SuppressWarnings("unchecked")
    @Override
    public void call(Iterator<Sequence<T>> sequenceIterator) throws Exception {
        /**
         * first we initialize
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

        if (shallowVocabCache == null)
            shallowVocabCache = vocabCacheBroadcast.getValue();

        if (elementsLearningAlgorithm == null && vectorsConfiguration.getElementsLearningAlgorithm() != null) {
            // TODO: do ELA initialization
            try {
                elementsLearningAlgorithm = (SparkElementsLearningAlgorithm) Class
                                .forName(vectorsConfiguration.getElementsLearningAlgorithm()).newInstance();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (elementsLearningAlgorithm != null)
            elementsLearningAlgorithm.configure(shallowVocabCache, null, vectorsConfiguration);

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
        if (sequenceLearningAlgorithm != null)
            sequenceLearningAlgorithm.configure(shallowVocabCache, null, vectorsConfiguration);

        if (elementsLearningAlgorithm == null && sequenceLearningAlgorithm == null) {
            throw new ND4JIllegalStateException("No LearningAlgorithms specified!");
        }


        List<Sequence<ShallowSequenceElement>> sequences = new ArrayList<>();

        // now we roll throw Sequences and prepare/convert/"learn" them
        while (sequenceIterator.hasNext()) {
            Sequence<T> sequence = sequenceIterator.next();

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

            sequences.add(mergedSequence);
            if (sequences.size() >= 8) {
                trainAllAtOnce(sequences);
                sequences.clear();
            }
        }

        if (!sequences.isEmpty()) {
            // finishing training round, to make sure we don't have trails
            trainAllAtOnce(sequences);
            sequences.clear();
        }
    }


    protected void trainAllAtOnce(List<Sequence<ShallowSequenceElement>> sequences) {
        Frame bigFrame = new Frame(BasicSequenceProvider.getInstance().getNextValue());

        for (Sequence<ShallowSequenceElement> sequence : sequences) {
            Frame frame = elementsLearningAlgorithm.frameSequence(sequence, new AtomicLong(119L), 25e-3f);
            bigFrame.stackMessages(frame.getMessages());
        }

        if (bigFrame.size() > 0)
            paramServer.execDistributed(bigFrame);
    }
}
