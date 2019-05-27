package org.nd4j.linalg.schedule;

/**
 * A "Wrapper" schedule that ramps up from {@code 1/numIter * baseLR} to {@code baseLR} over numIter iterations.
 * The base learning rate is determined by the underlying ISchedule, as a function of time.
 * This can be used to provide a slow start, for use cases such as transfer learning.
 *
 * @author Alex Black
 */
public class RampSchedule implements ISchedule {

    protected final ISchedule baseSchedule;
    protected final int numIter;

    public RampSchedule(ISchedule baseSchedule, int numIter){
        this.baseSchedule = baseSchedule;
        this.numIter = numIter;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        double base = baseSchedule.valueAt(iteration, epoch);
        if(iteration >= numIter - 1){
            return base;
        }
        double frac = (iteration+1) / (double)numIter;
        return frac * base;
    }

    @Override
    public ISchedule clone() {
        return new RampSchedule(baseSchedule.clone(), numIter);
    }
}
