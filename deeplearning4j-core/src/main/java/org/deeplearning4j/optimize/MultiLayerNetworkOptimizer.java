package org.deeplearning4j.optimize;

import java.io.Serializable;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.gradient.LogisticRegressionGradient;
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
public class MultiLayerNetworkOptimizer implements Optimizable.ByGradientValue,Serializable,OptimizableByGradientValueMatrix {

	private static final long serialVersionUID = -3012638773299331828L;

	protected BaseMultiLayerNetwork network;

	private static Logger log = LoggerFactory.getLogger(MultiLayerNetworkOptimizer.class);
	private double lr;

	public MultiLayerNetworkOptimizer(BaseMultiLayerNetwork network,double lr) {
		this.network = network;
		this.lr = lr;
	}



	public void optimize(DoubleMatrix labels,double lr,int epochs) {
		network.getLogLayer().setLabels(labels);
	
		DoubleMatrix train = sampleHiddenGivenVisible();
		
		if(!network.isForceNumEpochs()) {
			network.getLogLayer().trainTillConvergence(train,labels,lr,epochs);

			if(network.isShouldBackProp())
				network.backProp(lr, epochs);

		}
		
		else {
			log.info("Training for " + epochs + " epochs");
			for(int i = 0; i < epochs; i++) {
				network.getLogLayer().train(train, labels,lr);
			}
			

			if(network.isShouldBackProp())
				network.backProp(lr, epochs);

		}
	


	}



	private DoubleMatrix sampleHiddenGivenVisible() {
		return network.getSigmoidLayers()[network.getnLayers() - 1].sampleHiddenGivenVisible();
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
		LogisticRegressionGradient gradient = network.getLogLayer().getGradient(lr);
		
		DoubleMatrix weightGradient = gradient.getwGradient();
		DoubleMatrix biasGradient = gradient.getbGradient();
		
		int idx = 0;
		
		for(int i = 0; i < weightGradient.length; i++)
			buffer[idx++] = weightGradient.get(i);
		for(int i = 0; i < biasGradient.length; i++)
			buffer[idx++] = biasGradient.get(i);

	}



	@Override
	public double getValue() {
		return network.negativeLogLikelihood();
	}



	@Override
	public DoubleMatrix getParameters() {
		double[] d = new double[getNumParameters()];
		this.getParameters(d);
		return new DoubleMatrix(d);
	}



	@Override
	public void setParameters(DoubleMatrix params) {
		this.setParameters(params.toArray());
	}



	@Override
	public DoubleMatrix getValueGradient() {
		double[] buffer = new double[getNumParameters()];
		getValueGradient(buffer);
		return new DoubleMatrix(buffer);
	}


}
