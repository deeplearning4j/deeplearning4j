package org.deeplearning4j.scaleout.aggregator;



import org.deeplearning4j.scaleout.job.Job;


/**
 * Parameter averaging algorithm.
 * It handles summing and averaging over all of the results
 * accumulated so far
 */
public class DeepLearningAccumulator {

    private Job averaged = null;
    private int numWorkers;


    public DeepLearningAccumulator(int numWorkers) {
        this.numWorkers = numWorkers;
    }

    /**
     * Averages the results of the network so far
     * @param toAccumulate the network to average in
     */
    public void accumulate(Job toAccumulate) {


	}

    /**
     * The averaged network
     * @return the averaged network
     */
	public Job averaged() {
		return averaged;
	}

}
