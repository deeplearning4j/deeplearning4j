package org.deeplearning4j.nn.gradient;

import java.io.Serializable;

import org.jblas.DoubleMatrix;

public class LogisticRegressionGradient implements Serializable {

	
	private static final long serialVersionUID = -2843336269630396562L;
	private DoubleMatrix wGradient;
	private DoubleMatrix bGradient;
	
	
	
	/**
	 * Divies the gradient by the given number (used in averaging)
	 * @param num the number to divide by
	 */
	public void div(int num) {
		wGradient.divi(num);
		bGradient.divi(num);
	}
	
	/**
	 * Sums this gradient with the given one
	 * @param gradient the gradient to add
	 */
	public void add(LogisticRegressionGradient gradient) {
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
		return bGradient.columnMeans();
	}
	public void setbGradient(DoubleMatrix bGradient) {
		this.bGradient = bGradient;
	}
	
	

}
