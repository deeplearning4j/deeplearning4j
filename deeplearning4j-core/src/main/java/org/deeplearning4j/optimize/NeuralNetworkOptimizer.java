package org.deeplearning4j.optimize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.DeepLearningGradientAscent;
import org.deeplearning4j.util.OptimizerMatrix;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;

/**
 * Performs basic beam search based on the network's loss function
 * @author Adam Gibson
 *
 */
public abstract class NeuralNetworkOptimizer implements Optimizable.ByGradientValue,OptimizableByGradientValueMatrix,Serializable,NeuralNetEpochListener {




	public NeuralNetworkOptimizer(BaseNeuralNetwork network,double lr,Object[] trainingParams) {
		this.network = network;
		this.lr = lr;
		this.extraParams = trainingParams;
		if(network.useAdaGrad)
			network.resetAdaGrad(lr);
	}


	private static final long serialVersionUID = 4455143696487934647L;
	protected BaseNeuralNetwork network;
	protected double lr;
	protected Object[] extraParams;
	protected double tolerance = 0.00001;
	protected static Logger log = LoggerFactory.getLogger(NeuralNetworkOptimizer.class);
	protected List<Double> errors = new ArrayList<Double>();
	protected double minLearningRate = 0.001;
	protected transient OptimizerMatrix opt;

	public void train(DoubleMatrix x) {
		if(opt == null)
			opt = new VectorizedNonZeroStoppingConjugateGradient(this,this);
		//opt.setTolerance(tolerance);
		int epochs = (int) extraParams[2];
		opt.optimize(epochs);


	}
	@Override
	public void epochDone(int epoch) {
		int plotEpochs = network.getRenderEpochs();
		if(plotEpochs <= 0)
			return;
		if(epoch % plotEpochs == 0 || epoch == 0) {
			NeuralNetPlotter plotter = new NeuralNetPlotter();
			plotter.plotNetworkGradient(network,network.getGradient(extraParams));
		}
	}

	public List<Double> getErrors() {
		return errors;
	}


	@Override
	public int getNumParameters() {
		return network.W.length + network.hBias.length + network.vBias.length;
	}


	@Override
	public void getParameters(double[] buffer) {
		/*
		 * If we think of the parameters of the model (W,vB,hB)
		 * as a solid line for the optimizer, we get the following:
		 * 
		 */
		for(int i = 0; i < buffer.length; i++)
			buffer[i] = getParameter(i);
	}


	@Override
	public double getParameter(int index) {
		//beyond weight matrix
		if(index >= network.W.length) {
			int i = getAdjustedIndex(index);
			//beyond visible bias
			if(index >= network.vBias.length + network.W.length) {
				return network.hBias.get(i);
			}
			else
				return network.vBias.get(i);

		}
		else 
			return network.W.get(index);



	}


	@Override
	public void setParameters(double[] params) {
		/*
		 * If we think of the parameters of the model (W,vB,hB)
		 * as a solid line for the optimizer, we get the following:
		 * 
		 */
		for(int i = 0; i < params.length; i++)
			setParameter(i,params[i]);

	}


	@Override
	public void setParameter(int index, double value) {
		//beyond weight matrix
		if(index >= network.W.length) {
			//beyond visible bias
			if(index >= network.vBias.length + network.W.length)  {
				int i = getAdjustedIndex(index);
				network.hBias.put(i, value);
			}
			else {
				int i = getAdjustedIndex(index);
				network.vBias.put(i,value);

			}

		}
		else {
			network.W.put(index,value);
		}
	}


	private int getAdjustedIndex(int index) {
		int wLength = network.W.length;
		int vBiasLength = network.vBias.length;
		if(index < wLength)
			return index;
		else if(index >= wLength + vBiasLength) {
			int hIndex = index - wLength - vBiasLength;
			return hIndex;
		}
		else {
			int vIndex = index - wLength;
			return vIndex;
		}
	}

	
	

	@Override
	public DoubleMatrix getParameters() {
		double[] params = new double[getNumParameters()];
		this.getParameters(params);
		return new DoubleMatrix(params);
	}
	@Override
	public void setParameters(DoubleMatrix params) {
		setParameters(params.toArray());
	}
	@Override
	public DoubleMatrix getValueGradient() {
		double[] d = new double[this.getNumParameters()];
		getValueGradient(d);
		return new DoubleMatrix(d);
	}
	@Override
	public abstract void getValueGradient(double[] buffer);


	@Override
	public double getValue() {
		return -network.getReConstructionCrossEntropy();
	}



}
