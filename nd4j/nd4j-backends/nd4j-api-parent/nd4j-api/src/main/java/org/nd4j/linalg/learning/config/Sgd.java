package org.nd4j.linalg.learning.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.learning.SgdUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * SGD updater applies a learning rate only
 * @author Adam Gibson
 */
@Data
@EqualsAndHashCode
@Builder(builderClassName = "Builder")
public class Sgd implements IUpdater {
    public static final double DEFAULT_SGD_LR = 1e-3;

    @lombok.Builder.Default private double learningRate = DEFAULT_SGD_LR;
    private ISchedule learningRateSchedule;

    public Sgd(){
        this(DEFAULT_SGD_LR, null);
    }

    public Sgd(double learningRate){
        this(learningRate, null);
    }

    public Sgd(ISchedule learningRateSchedule){
        this(Double.NaN, learningRateSchedule);
    }

    private Sgd(@JsonProperty("learningRate") double learningRate,
                @JsonProperty("learningRateSchedule") ISchedule learningRateSchedule){
        this.learningRate = learningRate;
        this.learningRateSchedule = learningRateSchedule;
    }

    @Override
    public long stateSize(long numParams) {
        return 0;
    }

    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initializeViewArray) {
        if (viewArray != null) {
            throw new IllegalStateException("View arrays are not supported/required for SGD updater");
        }
        return new SgdUpdater(this);
    }

    @Override
    public Sgd clone() {
        return new Sgd(learningRate, learningRateSchedule);
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
