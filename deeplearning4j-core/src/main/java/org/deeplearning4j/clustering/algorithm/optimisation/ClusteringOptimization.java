package org.deeplearning4j.clustering.algorithm.optimisation;

public class ClusteringOptimization {

	private ClusteringOptimizationType	type;
	private double						value;

	

	public ClusteringOptimization(ClusteringOptimizationType type, double value) {
		super();
		this.type = type;
		this.value = value;
	}

	public ClusteringOptimizationType getType() {
		return type;
	}

	public void setType(ClusteringOptimizationType type) {
		this.type = type;
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

}
