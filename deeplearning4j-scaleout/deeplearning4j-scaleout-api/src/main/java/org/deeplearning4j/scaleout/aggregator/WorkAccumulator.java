package org.deeplearning4j.scaleout.aggregator;



import org.deeplearning4j.scaleout.job.Job;


/**
 * Parameter averaging algorithm.
 * It handles summing and averaging over all of the results
 * accumulated so far
 */
public abstract class WorkAccumulator implements JobAggregator {

    private Job averaged = null;
    protected double seenSoFar = 0.0;




    protected Job empty() {
        return new Job(null,"");
    }

    /**
     * Averages the results of the network so far
     * @param toAccumulate the network to average in
     */
    public abstract void accumulate(Job toAccumulate);



    /**
     * The averaged network
     * @return the averaged network
     */
    public Job averaged() {
        return averaged;
    }

}
