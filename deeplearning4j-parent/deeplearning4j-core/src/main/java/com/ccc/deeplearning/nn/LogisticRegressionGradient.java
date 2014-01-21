package com.ccc.deeplearning.nn;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class LogisticRegressionGradient implements Serializable {

	
	private static final long serialVersionUID = -2843336269630396562L;
	private DoubleMatrix wGradient;
	private DoubleMatrix bGradient;
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
		LogisticRegressionGradient other = (LogisticRegressionGradient) obj;
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
	public LogisticRegressionGradient(DoubleMatrix wGradient,
			DoubleMatrix bGradient) {
		super();
		this.wGradient = wGradient;
		this.bGradient = bGradient;
	}
	public DoubleMatrix getwGradient() {
		return wGradient;
	}
	public void setwGradient(DoubleMatrix wGradient) {
		this.wGradient = wGradient;
	}
	public DoubleMatrix getbGradient() {
		return bGradient;
	}
	public void setbGradient(DoubleMatrix bGradient) {
		this.bGradient = bGradient;
	}
	
	

}
