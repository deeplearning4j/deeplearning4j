package org.deeplearning4j.nn.conf.schedule;

import java.io.Serializable;

/**
 * ISchedule: a general purpose interface for getting values according to some schedule.
 * Used for implementing learning rate, dropout and momentum schedules - and in principle, any univariate (double)
 * value that deponds on the current iteration and epochs numbers.
 *
 * @author Alex Black
 */
public interface ISchedule extends Serializable, Cloneable {

    double valueAt(double currentValue, int iteration, int epoch);

    ISchedule clone();

}
