package com.ccc.deeplearning.optimize;

import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.deeplearning.util.MatrixUtil;

public class MultiLayerNetworkOptimizer {

	private BaseMultiLayerNetwork network;

	private static Logger log = LoggerFactory.getLogger(MultiLayerNetworkOptimizer.class);
	private double errorTolerance = 0.0000001;
	private List<Double> errors = new ArrayList<Double>();


	public MultiLayerNetworkOptimizer(BaseMultiLayerNetwork network) {
		this.network = network;
	}



	public void optimize(DoubleMatrix labels,double lr) {
		MatrixUtil.ensureValidOutcomeMatrix(labels);
		//sample from the final layer in the network and train on the result
		DoubleMatrix layerInput = network.sigmoidLayers[network.sigmoidLayers.length - 1].sample_h_given_v();
		network.logLayer.input = layerInput;
		network.logLayer.labels = labels;
		double cost = network.negativeLogLikelihood();
		boolean done = false;
		while(!done) {
			DoubleMatrix W = network.logLayer.W.dup();
			network.logLayer.train(layerInput, labels, lr);
			lr *= network.learningRateUpdate;
			double currCost = network.negativeLogLikelihood();
			if(currCost <= cost) {
				double diff = Math.abs(cost - currCost);
				if(diff <= errorTolerance) {
					done = true;
					log.info("Converged on finetuning at " + cost);
					break;
				}
				else
					cost = currCost;
				errors.add(cost);
				log.info("Found new log likelihood " + cost);
			}

			else if(currCost > cost) {
				done = true;
				network.logLayer.W = W;
				log.info("Converged on finetuning at " + cost + " due to a higher cost coming out than " + currCost);
				break;
			}
		}
	}



	public List<Double> getErrors() {
		return errors;
	}
	
	
}
