package com.ccc.deeplearning.nn;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class NeuralNetworkGradient implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 5611230066214840732L;
	private DoubleMatrix wGradient;
	private DoubleMatrix vBiasGradient;
	private DoubleMatrix hBiasGradient;
	
	
	
	
	
	
	
	
	public DoubleMatrix getwGradient() {
		return wGradient;
	}
	public void setwGradient(DoubleMatrix wGradient) {
		this.wGradient = wGradient;
	}
	public DoubleMatrix getvBiasGradient() {
		return vBiasGradient;
	}
	public void setvBiasGradient(DoubleMatrix vBiasGradient) {
		this.vBiasGradient = vBiasGradient;
	}
	public DoubleMatrix gethBiasGradient() {
		return hBiasGradient;
	}
	public void sethBiasGradient(DoubleMatrix hBiasGradient) {
		this.hBiasGradient = hBiasGradient;
	}
	public NeuralNetworkGradient(DoubleMatrix wGradient,
			DoubleMatrix vBiasGradient, DoubleMatrix hBiasGradient) {
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
	
	

}
