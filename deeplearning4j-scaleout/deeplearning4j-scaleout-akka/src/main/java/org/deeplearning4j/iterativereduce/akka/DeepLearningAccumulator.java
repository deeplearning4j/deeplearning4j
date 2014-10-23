package org.deeplearning4j.iterativereduce.akka;



import org.deeplearning4j.nn.BaseMultiLayerNetwork;


/**
 * Parameter averaging algorithm.
 * It handles summing and averaging over all of the results
 * accumulated so far
 */
public class DeepLearningAccumulator {

    private BaseMultiLayerNetwork averaged = null;
    private int numWorkers;


    public DeepLearningAccumulator(int numWorkers) {
        this.numWorkers = numWorkers;
    }

    /**
     * Averages the results of the network so far
     * @param toAccumulate the network to average in
     */
    public void accumulate(BaseMultiLayerNetwork toAccumulate) {
		if(averaged == null)
            this.averaged = toAccumulate;
        else
            averaged.merge(toAccumulate,numWorkers);

	}

    /**
     * The averaged network
     * @return the averaged network
     */
	public BaseMultiLayerNetwork averaged() {
		return averaged;
	}

}
