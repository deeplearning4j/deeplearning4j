package org.deeplearning4j.optimize;

import static org.deeplearning4j.util.MatrixUtil.softmax;

import java.io.Serializable;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.optimize.Optimizable;



/**
 * Optimizes the logistic layer for finetuning
 * a multi layer network. This is meant to be used
 * after pretraining.
 * @author Adam Gibson
 *
 */
public class MultiLayerNetworkOptimizer implements Optimizable.ByGradientValue,Serializable {

	private static final long serialVersionUID = -3012638773299331828L;

	protected BaseMultiLayerNetwork network;

	private static Logger log = LoggerFactory.getLogger(MultiLayerNetworkOptimizer.class);
	private double lr;

	public MultiLayerNetworkOptimizer(BaseMultiLayerNetwork network,double lr) {
		this.network = network;
		this.lr = lr;
	}



	public void optimize(DoubleMatrix labels,double lr,int epochs) {
		network.feedForward(network.getInput());
		//sample from the final layer in the network and train on the result
		DoubleMatrix layerInput = network.getSigmoidLayers()[network.getSigmoidLayers().length - 1].sample_h_given_v();
		network.getLogLayer().setInput(layerInput);
		network.getLogLayer().setLabels(labels);
		
		
		network.resetAdaGrad(lr);
		
		if(layerInput.rows != labels.rows) {
			throw new IllegalStateException("Labels not equal to input");
		}
		
		
		if(!network.isForceNumEpochs()) 
			network.getLogLayer().trainTillConvergence(lr,epochs);
		
		else {
			log.info("Training for " + epochs + " epochs");
			for(int i = 0; i < epochs; i++) {
				network.getLogLayer().train(layerInput, labels,lr);
			}
		}
		
		if(network.isShouldBackProp())
			network.backProp(lr, epochs);



	}








	@Override
	public int getNumParameters() {
		return network.getLogLayer().getW().length + network.getLogLayer().getB().length;
	}



	@Override
	public void getParameters(double[] buffer) {
		int idx = 0;
		for(int i = 0; i < network.getLogLayer().getW().length; i++) {
			buffer[idx++] = network.getLogLayer().getW().get(i);
			
		}
		for(int i = 0; i < network.getLogLayer().getB().length; i++) {
			buffer[idx++] = network.getLogLayer().getB().get(i);
		}
	}



	@Override
	public double getParameter(int index) {
		if(index >= network.getLogLayer().getW().length) {
			int i = index - network.getLogLayer().getB().length;
			return network.getLogLayer().getB().get(i);
		}
		else
			return network.getLogLayer().getW().get(index);
	}



	@Override
	public void setParameters(double[] params) {
		int idx = 0;
		for(int i = 0; i < network.getLogLayer().getW().length; i++) {
			network.getLogLayer().getW().put(i,params[idx++]);
		}
		for(int i = 0; i < network.getLogLayer().getB().length; i++) {
			network.getLogLayer().getB().put(i,params[idx++]);
		}
	}



	@Override
	public void setParameter(int index, double value) {
		if(index >= network.getLogLayer().getW().length) {
			int i = index - network.getLogLayer().getB().length;
			network.getLogLayer().getB().put(i,value);
		}
		else
			network.getLogLayer().getW().put(index,value);
	}



	@Override
	public void getValueGradient(double[] buffer) {
		DoubleMatrix p_y_given_x = softmax(network.getLogLayer().getInput().mmul(network.getLogLayer().getW()).addRowVector(network.getLogLayer().getB()));
		DoubleMatrix dy = network.getLogLayer().getLabels().sub(p_y_given_x);

		int idx = 0;
		DoubleMatrix weightGradient = network.getLogLayer().getInput().transpose().mmul(dy).mul(lr);
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
