package org.deeplearning4j.nn.gradient;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;


public class OutputLayerGradient implements Serializable {

	
	private static final long serialVersionUID = -2843336269630396562L;
	private INDArray wGradient;
	private INDArray bGradient;
	
	
	
	/**
	 * Divies the gradient by the given number (used in averaging)
	 * @param num the number to divide by
	 */
	public void div(int num) {
		wGradient.divi(Nd4j.scalar(num));
		bGradient.divi(Nd4j.scalar(num));
	}
	
	/**
	 * Sums this gradient with the given one
	 * @param gradient the gradient to add
	 */
	public void add(OutputLayerGradient gradient) {
		wGradient.addi(gradient.getwGradient());
		bGradient.addi(gradient.getbGradient());
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((bGradient == null) ? 0 : bGradient.hashCode());
		result = prime * result
				+ ((wGradient == null) ? 0 : wGradient.hashCode());
		return result;
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		OutputLayerGradient other = (OutputLayerGradient) obj;
		if (bGradient == null) {
			if (other.bGradient != null)
				return false;
		} else if (!bGradient.equals(other.bGradient))
			return false;
		if (wGradient == null) {
			if (other.wGradient != null)
				return false;
		} else if (!wGradient.equals(other.wGradient))
			return false;
		return true;
	}
	public OutputLayerGradient(INDArray wGradient,
                               INDArray bGradient) {
		super();
		this.wGradient = wGradient;
		this.bGradient = bGradient;
	}

    public INDArray getwGradient() {
		return wGradient;
	}

    public void setwGradient(INDArray wGradient) {
		this.wGradient = wGradient;
	}

    public INDArray getbGradient() {
		return bGradient;
	}

    public void setbGradient(INDArray bGradient) {
		this.bGradient = bGradient;
	}
	
	

}
