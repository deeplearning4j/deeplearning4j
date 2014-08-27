package org.deeplearning4j.optimize.optimizers.rbm;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.optimize.optimizers.NeuralNetworkOptimizer;

/**
 * Optimizes an RBM.
 * Handles dissemination of a parameter vector 
 * via the weights, hidden bias, and visible bias
 * 
 * @author Adam Gibson
 * 
 * @see{RBM}
 *
 */
public class RBMOptimizer extends NeuralNetworkOptimizer {

	
	private static final long serialVersionUID = 3676032651650426749L;
	protected int k = -1;
	protected int numTimesIterated = 0;
	
	public RBMOptimizer(BaseNeuralNetwork network,float lr, Object[] trainingParams,OptimizationAlgorithm optimizationAlgorithm,LossFunction lossFunction) {
		super(network,lr,trainingParams,optimizationAlgorithm,lossFunction);
	}



    @Override
    public INDArray getValueGradient(int iteration) {
        int k = (int) extraParams[0];

        numTimesIterated++;
        //adaptive k based on the number of iterations.
        //typically over time, you want to increase k.
        if(this.k <= 0)
            this.k = k;
        if(numTimesIterated % 10 == 0) {
            this.k++;
        }


        return super.getValueGradient(iteration);
    }





}
