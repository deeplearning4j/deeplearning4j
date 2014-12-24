package org.deeplearning4j.nn.gradient;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

import org.deeplearning4j.nn.api.Persistable;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Gradient for a whole multi layer network
 * @author Adam Gibson
 *
 */
public class MultiLayerGradient implements Persistable,Cloneable,Gradient {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5262146791172613616L;
	private List<NeuralNetworkGradient> gradients;
	private OutputLayerGradient logRegGradient;
	
	
	
	
	public MultiLayerGradient(List<NeuralNetworkGradient> gradients,
			OutputLayerGradient logRegGradient) {
		super();
		this.gradients = gradients;
		this.logRegGradient = logRegGradient;
	}

	@Override
	public void write(OutputStream os) {
		SerializationUtils.writeObject(this, os);
	}

	@Override
	public void load(InputStream is) {
		MultiLayerGradient read = SerializationUtils.readObject(is);
		this.gradients = read.gradients;
		this.logRegGradient = read.logRegGradient;
	}

	public void div(int num) {
		for(NeuralNetworkGradient g : gradients)
			g.div(num);
		
			
	}
	
	
	@Override
	public MultiLayerGradient clone() {
		return org.apache.commons.lang3.SerializationUtils.clone(this);
	}

	public void addGradient(MultiLayerGradient other) {
		for(int i = 0;i < gradients.size(); i++) {
			gradients.get(i).add(other.getGradients().get(i));
		}
		
		logRegGradient.add(other.getLogRegGradient());
	}
	
	public  List<NeuralNetworkGradient> getGradients() {
		return gradients;
	}

	public  void setGradients(List<NeuralNetworkGradient> gradients) {
		this.gradients = gradients;
	}

	public OutputLayerGradient getLogRegGradient() {
		return logRegGradient;
	}

	public  void setLogRegGradient(
			OutputLayerGradient logRegGradient) {
		this.logRegGradient = logRegGradient;
	}


	@Override
	public INDArray gradient() {
		INDArray[] gradients = new INDArray[this.gradients.size() + 1];
		for(int i = 0; i < gradients.length; i++)
			gradients[i] = this.getGradients().get(i).gradient();
		gradients[gradients.length - 1] = logRegGradient.gradient();
		return Nd4j.concat(0,gradients);
	}

	@Override
	public void clear() {
        gradients.clear();
		logRegGradient = null;
	}
}
