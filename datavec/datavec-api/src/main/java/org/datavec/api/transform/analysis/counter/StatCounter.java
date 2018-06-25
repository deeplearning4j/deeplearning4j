package org.datavec.api.transform.analysis.counter;

import java.io.Serializable;

public class StatCounter implements Serializable {

    private long count = 0;
    private double runningMean;
    private double runningM2;   // Running variance numerator (sum of (x - mean)^2)
    private double max = -Double.MAX_VALUE;
    private double min = Double.MAX_VALUE;


    public double getMean(){
        return runningMean;
    }

    public double getSum(){
        return runningMean * count;
    }

    public double getMin(){
        return min;
    }

    public double getMax(){
        return max;
    }

    public long getCount(){
        return count;
    }

    public double getVariance(boolean population){
        long divisor = (population ? count : count-1);
        if( (population && count == 0) || (!population && count == 1)){
            return Double.NaN;
        }
        return runningM2 / divisor;
    }

    public double getStddev(boolean population){
        return Math.sqrt(getVariance(population));
    }

    public void add(double x){
        double d = x - runningMean;
        count++;
        runningMean += (d / count);
        runningM2 += (d * (x - runningMean));
        max = Math.max(max, x);
        min = Math.min(min, x);
    }

    public StatCounter merge(StatCounter o){
        if(o == null || o.count == 0){
            return this;
        }
        if(o == this){
            return merge(o.clone());
        }
        if(this.count == 0){
            count = o.count;
            runningMean = o.runningMean;
            runningMean = o.runningM2;
            max = o.max;
            min = o.min;
        } else {
            min = Math.min(min, o.min);
            max = Math.max(max, o.max);

            double d = o.runningMean - runningMean;
            if (o.count * 10 < count) {
                runningMean = runningMean + (d * o.count) / (count + o.count);
            } else if (count * 10 < o.count) {
                runningMean = o.runningMean - (d * count) / (count + o.count);
            } else {
                runningMean = (runningMean * count + o.runningMean * o.count) / (count + o.count);
            }
            runningM2 += o.runningM2 + (d * d * count * o.runningM2) / (count + o.count);
            count += o.count;
        }

        return this;
    }

    public StatCounter clone(){
        StatCounter ret = new StatCounter();
        ret.count = count;
        ret.runningMean = runningMean;
        ret.runningM2 = runningM2;
        ret.max = max;
        ret.min = min;
        return ret;
    }
}
