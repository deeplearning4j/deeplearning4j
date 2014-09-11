package org.deeplearning4j.optimize.optimizers.rbm;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.BaseNeuralNetwork;

import org.deeplearning4j.nn.api.NeuralNetwork;
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
	
	public RBMOptimizer(BaseNeuralNetwork network,float lr, Object[] trainingParams,NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm,LossFunctions.LossFunction lossFunction) {
		super(network,lr,trainingParams,optimizationAlgorithm,lossFunction);
        if(extraParams.length == 1 && extraParams[0] == null)
            extraParams[0] = 1;
	}



    @Override
    public INDArray getValueGradient(int iteration) {
        int k = extraParams != null  && extraParams.length < 1 ? 1 : (int) extraParams[0];

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
