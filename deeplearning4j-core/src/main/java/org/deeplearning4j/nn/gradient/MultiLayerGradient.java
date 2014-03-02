package org.deeplearning4j.nn.gradient;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;

import org.deeplearning4j.nn.Persistable;
import org.deeplearning4j.util.SerializationUtils;

/**
 * Gradient for a whole multi layer network
 * @author Adam Gibson
 *
 */
public class MultiLayerGradient implements Persistable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5262146791172613616L;
	private List<NeuralNetworkGradient> gradients;
	private LogisticRegressionGradient logRegGradient;
	
	
	
	
	public MultiLayerGradient(List<NeuralNetworkGradient> gradients,
			LogisticRegressionGradient logRegGradient) {
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

	public synchronized List<NeuralNetworkGradient> getGradients() {
		return gradients;
	}

	public synchronized void setGradients(List<NeuralNetworkGradient> gradients) {
		this.gradients = gradients;
	}

	public synchronized LogisticRegressionGradient getLogRegGradient() {
		return logRegGradient;
	}

	public synchronized void setLogRegGradient(
			LogisticRegressionGradient logRegGradient) {
		this.logRegGradient = logRegGradient;
	}

	
}
