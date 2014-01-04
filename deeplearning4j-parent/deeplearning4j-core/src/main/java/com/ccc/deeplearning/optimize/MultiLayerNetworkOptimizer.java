package com.ccc.deeplearning.optimize;

import static com.ccc.deeplearning.util.MatrixUtil.softmax;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.optimize.ConjugateGradient;
import cc.mallet.optimize.InvalidOptimizableException;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.OptimizationException;
import cc.mallet.optimize.Optimizer;

import com.ccc.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.deeplearning.util.MatrixUtil;


/**
 * Optimizes the logistic layer for finetuning
 * a multi layer network. This is meant to be used
 * after pretraining.
 * @author Adam Gibson
 *
 */
public class MultiLayerNetworkOptimizer implements Optimizable.ByGradientValue,Serializable {

	private static final long serialVersionUID = -3012638773299331828L;

	private BaseMultiLayerNetwork network;

	private static Logger log = LoggerFactory.getLogger(MultiLayerNetworkOptimizer.class);
	private double lr;
	private transient Optimizer opt;

	public MultiLayerNetworkOptimizer(BaseMultiLayerNetwork network,double lr) {
		this.network = network;
		this.lr = lr;
	}



	public void optimize(DoubleMatrix labels,double lr,int epochs) {
		MatrixUtil.ensureValidOutcomeMatrix(labels);
		//sample from the final layer in the network and train on the result



		DoubleMatrix layerInput = network.sigmoidLayers[network.sigmoidLayers.length - 1].sample_h_given_v();
		network.logLayer.input = layerInput;
		network.logLayer.labels = labels;

		network.logLayer.train(layerInput, labels, lr);


		lr *= network.learningRateUpdate;
		opt  = new ConjugateGradient(this);

		boolean done = false;
		for(int i = 0; i < epochs; i++)
			while(!done) {
				try {
					done = opt.optimize();
				}
				catch(InvalidOptimizableException e) {
					network.logLayer.input = layerInput;
					network.logLayer.labels = labels;

					network.logLayer.train(layerInput, labels, lr);


					lr *= network.learningRateUpdate;

					log.warn("Invalid step taken; trying again. Error was ",e);
				}
				catch(OptimizationException e2) {
					network.logLayer.input = layerInput;
					network.logLayer.labels = labels;

					network.logLayer.train(layerInput, labels, lr);


					lr *= network.learningRateUpdate;
					log.warn("Invalid step taken; trying again. Error was ",e2);

				}

			}

	}


	@Override
	public int getNumParameters() {
		return network.logLayer.W.length + network.logLayer.b.length;
	}



	@Override
	public void getParameters(double[] buffer) {
		int idx = 0;
		for(int i = 0; i < network.logLayer.W.length; i++) {
			buffer[idx++] = network.logLayer.W.get(i);
		}
		for(int i = 0; i < network.logLayer.b.length; i++) {
			buffer[idx++] = network.logLayer.b.get(i);
		}
	}



	@Override
	public double getParameter(int index) {
		if(index >= network.logLayer.W.length) {
			int i = index - network.logLayer.b.length;
			return network.logLayer.b.get(i);
		}
		else
			return network.logLayer.W.get(index);
	}



	@Override
	public void setParameters(double[] params) {
		int idx = 0;
		for(int i = 0; i < network.logLayer.W.length; i++) {
			network.logLayer.W.put(i,params[idx++]);
		}
		for(int i = 0; i < network.logLayer.b.length; i++) {
			network.logLayer.b.put(i,params[idx++]);
		}
	}



	@Override
	public void setParameter(int index, double value) {
		if(index >= network.logLayer.W.length) {
			int i = index - network.logLayer.b.length;
			network.logLayer.b.put(i,value);
		}
		else
			network.logLayer.W.put(index,value);
	}



	@Override
	public void getValueGradient(double[] buffer) {
		DoubleMatrix p_y_given_x = softmax(network.logLayer.input.mmul(network.logLayer.W).addRowVector(network.logLayer.b));
		DoubleMatrix dy = network.logLayer.labels.sub(p_y_given_x);

		int idx = 0;
		DoubleMatrix weightGradient = network.logLayer.input.transpose().mmul(dy).mul(lr);
		DoubleMatrix biasGradient =  dy.columnMeans().mul(lr);
		for(int i = 0; i < weightGradient.length; i++)
			buffer[idx++] = weightGradient.get(i);
		for(int i = 0; i < biasGradient.length; i++)
			buffer[idx++] = biasGradient.get(i);

	}



	@Override
	public double getValue() {
		return network.negativeLogLikelihood();
	}


}
