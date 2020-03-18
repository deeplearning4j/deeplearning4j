package org.deeplearning4j.rl4j.util;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.EpochStepCounter;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.filter.UniformSkippingFilter;
import org.deeplearning4j.rl4j.observation.transform.legacy.EncodableToINDArrayTransform;
import org.deeplearning4j.rl4j.observation.transform.legacy.EncodableToImageWritableTransform;
import org.deeplearning4j.rl4j.observation.transform.legacy.ImageWritableToINDArrayTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.HistoryMergeTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.SimpleNormalizationTransform;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;

public class LegacyMDPWrapper<O, A, AS extends ActionSpace<A>> implements MDP<Observation, A, AS> {

    @Getter
    private final MDP<O, A, AS> wrappedMDP;
    @Getter
    private final WrapperObservationSpace observationSpace;
    private final int[] shape;

    @Setter
    private TransformProcess transformProcess;

    @Getter(AccessLevel.PRIVATE)
    private IHistoryProcessor historyProcessor;

    private final EpochStepCounter epochStepCounter;

    private int skipFrame = 1;

    public LegacyMDPWrapper(MDP<O, A, AS> wrappedMDP, IHistoryProcessor historyProcessor, EpochStepCounter epochStepCounter) {
        this.wrappedMDP = wrappedMDP;
        this.shape = wrappedMDP.getObservationSpace().getShape();
        this.observationSpace = new WrapperObservationSpace(shape);
        this.historyProcessor = historyProcessor;
        this.epochStepCounter = epochStepCounter;

        setHistoryProcessor(historyProcessor);
    }

    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
        createTransformProcess();
    }

    private void createTransformProcess() {
        IHistoryProcessor historyProcessor = getHistoryProcessor();

        if(historyProcessor != null && shape.length == 3) {
            int skipFrame = historyProcessor.getConf().getSkipFrame();

            int finalHeight = historyProcessor.getConf().getCroppingHeight();
            int finalWidth = historyProcessor.getConf().getCroppingWidth();

            transformProcess = TransformProcess.builder()
                    .filter(new UniformSkippingFilter(skipFrame))
                    .transform("data", new EncodableToImageWritableTransform(shape[0], shape[1], shape[2]))
                    .transform("data", new MultiImageTransform(
                            new ResizeImageTransform(historyProcessor.getConf().getRescaledWidth(), historyProcessor.getConf().getRescaledHeight()),
                            new ColorConversionTransform(COLOR_BGR2GRAY),
                            new CropImageTransform(historyProcessor.getConf().getOffsetY(), historyProcessor.getConf().getOffsetX(), finalHeight, finalWidth)
                    ))
                    .transform("data", new ImageWritableToINDArrayTransform(finalHeight, finalWidth))
                    .transform("data", new SimpleNormalizationTransform(0.0, 255.0))
                    .transform("data", HistoryMergeTransform.builder()
                            .isFirstDimenstionBatch(true)
                            .build())
                    .build("data");
        }
        else {
            transformProcess = TransformProcess.builder()
                    .transform("data", new EncodableToINDArrayTransform(shape))
                    .build("data");
        }
    }

    @Override
    public AS getActionSpace() {
        return wrappedMDP.getActionSpace();
    }

    @Override
    public Observation reset() {
        transformProcess.reset();

        O rawResetResponse = wrappedMDP.reset();
        record(rawResetResponse);

        if(historyProcessor != null) {
            skipFrame = historyProcessor.getConf().getSkipFrame();
        }

        Map<String, Object> channelsData = buildChannelsData(rawResetResponse);
        return transformProcess.transform(channelsData, 0, false);
    }

    @Override
    public StepReply<Observation> step(A a) {
        IHistoryProcessor historyProcessor = getHistoryProcessor();

        StepReply<O> rawStepReply = wrappedMDP.step(a);
        INDArray rawObservation = getInput(rawStepReply.getObservation());

        if(historyProcessor != null) {
            historyProcessor.record(rawObservation);
        }

        int stepOfObservation = epochStepCounter.getCurrentEpochStep() + 1;

        Map<String, Object> channelsData = buildChannelsData(rawStepReply.getObservation());
        Observation observation =  transformProcess.transform(channelsData, stepOfObservation, rawStepReply.isDone());
        return new StepReply<Observation>(observation, rawStepReply.getReward(), rawStepReply.isDone(), rawStepReply.getInfo());
    }

    private void record(O obs) {
        INDArray rawObservation = getInput(obs);

        IHistoryProcessor historyProcessor = getHistoryProcessor();
        if(historyProcessor != null) {
            historyProcessor.record(rawObservation);
        }
    }

    private Map<String, Object> buildChannelsData(final O obs) {
        return new HashMap<String, Object>() {{
            put("data", obs);
        }};
    }

    @Override
    public void close() {
        wrappedMDP.close();
    }

    @Override
    public boolean isDone() {
        return wrappedMDP.isDone();
    }

    @Override
    public MDP<Observation, A, AS> newInstance() {
        return new LegacyMDPWrapper<O, A, AS>(wrappedMDP.newInstance(), historyProcessor, epochStepCounter);
    }

    private INDArray getInput(O obs) {
        INDArray arr = Nd4j.create(((Encodable)obs).toArray());
        int[] shape = observationSpace.getShape();
        if (shape.length == 1)
            return arr.reshape(new long[] {1, arr.length()});
        else
            return arr.reshape(shape);
    }

    public static class WrapperObservationSpace implements ObservationSpace<Observation> {

        @Getter
        private final int[] shape;

        public WrapperObservationSpace(int[] shape) {

            this.shape = shape;
        }

        @Override
        public String getName() {
            return null;
        }

        @Override
        public INDArray getLow() {
            return null;
        }

        @Override
        public INDArray getHigh() {
            return null;
        }
    }
}
