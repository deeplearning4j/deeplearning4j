package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.RmsPropUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * RMS Prop updates:
 * <p>
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
@Builder(builderClassName = "Builder")
public class RmsProp implements IUpdater {
    public static final double DEFAULT_RMSPROP_LEARNING_RATE = 1e-1;
    public static final double DEFAULT_RMSPROP_EPSILON = 1e-8;
    public static final double DEFAULT_RMSPROP_RMSDECAY = 0.95;

    @lombok.Builder.Default private double learningRate = DEFAULT_RMSPROP_LEARNING_RATE;
    private ISchedule learningRateSchedule;
    @lombok.Builder.Default private double rmsDecay = DEFAULT_RMSPROP_RMSDECAY;
    @lombok.Builder.Default private double epsilon = DEFAULT_RMSPROP_EPSILON;

    public RmsProp(){
        this(DEFAULT_RMSPROP_LEARNING_RATE, null, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(double learningRate){
        this(learningRate, null, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule, DEFAULT_RMSPROP_RMSDECAY, DEFAULT_RMSPROP_EPSILON);
    }

    public RmsProp(double learningRate, double rmsDecay, double epsilon){
        this(learningRate, null, rmsDecay, epsilon);
    }

    private RmsProp(@JsonProperty("learningRate") double learningRate,
                   @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule,
                   @JsonProperty("rmsDecay") double rmsDecay,
                   @JsonProperty("epsilon") double epsilon){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
        this.rmsDecay = rmsDecay;
        this.epsilon = epsilon;
    }

    @Override
    public long stateSize(long numParams) {
        return numParams;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        RmsPropUpdater u = new RmsPropUpdater(this);
        u.setStateViewArray(viewArray, viewArray.shape(), viewArray.ordering(), initializeViewArray);
        return u;
    }

    @Override
    public RmsProp clone() {
        return new RmsProp(learningRate, learningRateSchedule, rmsDecay, epsilon);
    }

    @Override
    public double getLearningRate(int iteration, int epoch){
        if(learningRateSchedule != null){
            return learningRateSchedule.valueAt(iteration, epoch);
        }
        return learningRate;
    }

    @Override
    public boolean hasLearningRate() {
        return true;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
        this.learningRate = lr;
        this.learningRateSchedule = lrSchedule;
    }

    //Partial builder implementation to give public no-arg constructor
    public static class Builder {
        public Builder(){ }
    }
}
