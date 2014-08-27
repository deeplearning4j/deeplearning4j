package org.deeplearning4j.nn.gradient;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.nn.api.Persistable;

/**
 * Represents the gradient for changing a neural network
 * @author Adam Gibson
 *
 */
public class NeuralNetworkGradient implements Serializable,Persistable {
	
	private static final long serialVersionUID = 5611230066214840732L;
	private INDArray wGradient;
	private INDArray vBiasGradient;
	private INDArray hBiasGradient;


	public INDArray getwGradient() {
		return wGradient;
	}
	public void setwGradient(INDArray wGradient) {
		this.wGradient = wGradient;
	}
	public INDArray getvBiasGradient() {
		return vBiasGradient;
	}
	public void setvBiasGradient(INDArray vBiasGradient) {
		this.vBiasGradient = vBiasGradient;
	}
	public INDArray gethBiasGradient() {
		return hBiasGradient;
	}
	public void sethBiasGradient(INDArray hBiasGradient) {
		this.hBiasGradient = hBiasGradient;
	}
	public NeuralNetworkGradient(INDArray wGradient,
			INDArray vBiasGradient, INDArray hBiasGradient) {
		super();
		this.wGradient = wGradient;
		this.vBiasGradient = vBiasGradient;
		this.hBiasGradient = hBiasGradient;
	}
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((hBiasGradient == null) ? 0 : hBiasGradient.hashCode());
		result = prime * result
				+ ((vBiasGradient == null) ? 0 : vBiasGradient.hashCode());
		result = prime * result
				+ ((wGradient == null) ? 0 : wGradient.hashCode());
		return result;
	}
	
	/**
	 * Adds the given gradient and this one together
	 * @param gradient the gradient to add
	 */
	public void add(NeuralNetworkGradient gradient) {
		wGradient.addi(gradient.getwGradient());
		vBiasGradient.addi(gradient.getvBiasGradient());
		hBiasGradient.addi(gradient.gethBiasGradient());
	}
	
	/**
	 * Divides the gradients by the given number
	 * @param num the number to divie by
	 */
	public void div(int num) {
		wGradient.divi(NDArrays.scalar(num));
		vBiasGradient.divi(NDArrays.scalar(num));
		hBiasGradient.divi(NDArrays.scalar(num));
	}
	
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		NeuralNetworkGradient other = (NeuralNetworkGradient) obj;
		if (hBiasGradient == null) {
			if (other.hBiasGradient != null)
				return false;
		} else if (!hBiasGradient.equals(other.hBiasGradient))
			return false;
		if (vBiasGradient == null) {
			if (other.vBiasGradient != null)
				return false;
		} else if (!vBiasGradient.equals(other.vBiasGradient))
			return false;
		if (wGradient == null) {
			if (other.wGradient != null)
				return false;
		} else if (!wGradient.equals(other.wGradient))
			return false;
		return true;
	}
	@Override
	public void write(OutputStream os) {
		ObjectOutputStream os2;
		try {
			os2 = new ObjectOutputStream(os);
			os2.writeObject(this);

		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}
	@Override
	public void load(InputStream is) {
		try {
			ObjectInputStream ois = new ObjectInputStream(is);
			NeuralNetworkGradient g = (NeuralNetworkGradient) ois.readObject();
			this.wGradient = g.wGradient;
			this.vBiasGradient = g.vBiasGradient;
			this.hBiasGradient = g.hBiasGradient;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

	}



}
