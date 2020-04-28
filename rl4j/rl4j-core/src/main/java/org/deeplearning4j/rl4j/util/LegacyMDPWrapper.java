package org.deeplearning4j.rl4j.util;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.transform.EncodableToINDArrayTransform;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.observation.transform.TransformProcess;
import org.deeplearning4j.rl4j.observation.transform.filter.UniformSkippingFilter;
import org.deeplearning4j.rl4j.observation.transform.legacy.EncodableToImageWritableTransform;
import org.deeplearning4j.rl4j.observation.transform.legacy.ImageWritableToINDArrayTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.HistoryMergeTransform;
import org.deeplearning4j.rl4j.observation.transform.operation.SimpleNormalizationTransform;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;

public class LegacyMDPWrapper<OBSERVATION extends Encodable, A, AS extends ActionSpace<A>> implements MDP<Observation, A, AS> {

    @Getter
    private final MDP<OBSERVATION, A, AS> wrappedMDP;
    @Getter
    private final WrapperObservationSpace observationSpace;
    private final int[] shape;

    @Setter
    private TransformProcess transformProcess;

    @Getter(AccessLevel.PRIVATE)
    private IHistoryProcessor historyProcessor;

    private int skipFrame = 1;
    private int steps = 0;


    public LegacyMDPWrapper(MDP<OBSERVATION, A, AS> wrappedMDP, IHistoryProcessor historyProcessor) {
        this.wrappedMDP = wrappedMDP;
        this.shape = wrappedMDP.getObservationSpace().getShape();
        this.observationSpace = new WrapperObservationSpace(shape);
        this.historyProcessor = historyProcessor;

        setHistoryProcessor(historyProcessor);
    }

    public void setHistoryProcessor(IHistoryProcessor historyProcessor) {
        this.historyProcessor = historyProcessor;
        createTransformProcess();
    }

    //TODO: this transform process should be decoupled from history processor and configured seperately by the end-user
    private void createTransformProcess() {
        IHistoryProcessor historyProcessor = getHistoryProcessor();

        if(historyProcessor != null && shape.length == 3) {
            int skipFrame = historyProcessor.getConf().getSkipFrame();
            int frameStackLength = historyProcessor.getConf().getHistoryLength();

            int height = shape[1];
            int width = shape[2];

            int cropBottom = height - historyProcessor.getConf().getCroppingHeight();
            int cropRight = width - historyProcessor.getConf().getCroppingWidth();

            transformProcess = TransformProcess.builder()
                    .filter(new UniformSkippingFilter(skipFrame))
                    .transform("data", new EncodableToImageWritableTransform())
                    .transform("data", new MultiImageTransform(
                            new CropImageTransform(historyProcessor.getConf().getOffsetY(), historyProcessor.getConf().getOffsetX(), cropBottom, cropRight),
                            new ResizeImageTransform(historyProcessor.getConf().getRescaledWidth(), historyProcessor.getConf().getRescaledHeight()),
                            new ColorConversionTransform(COLOR_BGR2GRAY)
                            //new ShowImageTransform("crop + resize + greyscale")
                    ))
                    .transform("data", new ImageWritableToINDArrayTransform())
                    .transform("data", new SimpleNormalizationTransform(0.0, 255.0))
                    .transform("data", HistoryMergeTransform.builder()
                            .isFirstDimenstionBatch(true)
                            .build(frameStackLength))
                    .build("data");
        }
        else {
            transformProcess = TransformProcess.builder()
                    .transform("data", new EncodableToINDArrayTransform())
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

        OBSERVATION rawResetResponse = wrappedMDP.reset();
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

        StepReply<OBSERVATION> rawStepReply = wrappedMDP.step(a);
        INDArray rawObservation = getInput(rawStepReply.getObservation());

        if(historyProcessor != null) {
            historyProcessor.record(rawObservation);
        }

        int stepOfObservation = steps++;

        Map<String, Object> channelsData = buildChannelsData(rawStepReply.getObservation());
        Observation observation =  transformProcess.transform(channelsData, stepOfObservation, rawStepReply.isDone());

        return new StepReply<Observation>(observation, rawStepReply.getReward(), rawStepReply.isDone(), rawStepReply.getInfo());
    }

    private void record(OBSERVATION obs) {
        INDArray rawObservation = getInput(obs);

        IHistoryProcessor historyProcessor = getHistoryProcessor();
        if(historyProcessor != null) {
            historyProcessor.record(rawObservation);
        }
    }

    private Map<String, Object> buildChannelsData(final OBSERVATION obs) {
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
        return new LegacyMDPWrapper<>(wrappedMDP.newInstance(), historyProcessor);
    }

    private INDArray getInput(OBSERVATION obs) {
        return obs.getData();
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
