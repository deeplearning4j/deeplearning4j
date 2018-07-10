package org.nd4j.linalg.schedule;

import lombok.Data;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Cycle schedule
 *
 *  Based on 1cycle schedule as proposed in https://arxiv.org/abs/1803.09820
 *
 *  Starts at initial learning rate, then linearly increases learning rate until max learning rate is reached,
 *  at that point the learning rate is decreased back to initial learning rate.
 *
 *  When cycleLength - annealingLength is reached, the annealing period starts, and the learning rate starts decaying
 *  below the initial learning rate.
 *
 *  The Learning rate curve Looks something like this:
 *
 * +-----------------------------------------+
 * |               XX                        |
 * |              XX XX                      |
 * |             XX   XX                     |
 * |            XX     XX                    |
 * |           XX       XX                   |
 * |          XX         XX                  |
 * |         XX           XX                 |
 * |        XX             XX                |
 * |       XX               XX               |
 * |      XX                 XX              |
 * |     XX                   XX             |
 * |    XX                     XX            |
 * |   XX                       XX           |
 * |  XX                          XXX        |
 * |                                XXX      |
 * |                                   XXX   |
 * |                                         |
 * +-----------------------------------------+
 *
 * @author Paul Dubs
 */
@Data
public class CycleSchedule implements ISchedule {

    private final ScheduleType scheduleType;
    private final double initialLearningRate;
    private final double maxLearningRate;
    private final int cycleLength;
    private final int annealingLength;
    private final int stepSize;
    private final double increment;
    private double annealingDecay;

    public CycleSchedule(@JsonProperty("scheduleType") ScheduleType scheduleType,
                         @JsonProperty("initialLearningRate") double initialLearningRate,
                         @JsonProperty("maxLearningRate") double maxLearningRate,
                         @JsonProperty("cycleLength") int cycleLength,
                         @JsonProperty("annealingLength") int annealingLength,
                         @JsonProperty("annealingDecay") double annealingDecay){
        this.scheduleType = scheduleType;
        this.initialLearningRate = initialLearningRate;
        this.maxLearningRate = maxLearningRate;
        this.cycleLength = cycleLength;
        this.annealingDecay = annealingDecay;
        this.annealingLength = annealingLength;

        stepSize = ((cycleLength - annealingLength) / 2);
        increment = (maxLearningRate - initialLearningRate) / stepSize;
    }

    public CycleSchedule(ScheduleType scheduleType,
                         double maxLearningRate,
                         int cycleLength){
        this(scheduleType, maxLearningRate * 0.1, maxLearningRate, cycleLength, (int) Math.round(cycleLength * 0.1), 0.1);
    }


    @Override
    public double valueAt(int iteration, int epoch) {
        double learningRate;
        final int positionInCycle = (scheduleType == ScheduleType.EPOCH ? epoch : iteration) % cycleLength;

        if(positionInCycle < stepSize){
            learningRate = initialLearningRate + increment * positionInCycle;
        }else if(positionInCycle < 2*stepSize){
            learningRate = maxLearningRate - increment * (positionInCycle - stepSize);
        }else {
            learningRate = initialLearningRate * Math.pow(annealingDecay, annealingLength - (cycleLength - positionInCycle));
        }

        return learningRate;
    }

    @Override
    public ISchedule clone() {
        return new CycleSchedule(scheduleType, initialLearningRate, maxLearningRate, cycleLength, annealingLength, annealingDecay);
    }
}
