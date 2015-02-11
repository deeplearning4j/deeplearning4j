package org.deeplearning4j.clustering.cluster.info;

import java.util.HashMap;
import java.util.Map;

public class ClusterInfo {

	private double				averagePointDistanceFromCenter;
	private double				totalPointDistanceFromCenter;
	private Map<String, Double>	pointDistancesFromCenter	= new HashMap<String, Double>();

	public double getAveragePointDistanceFromCenter() {
		return averagePointDistanceFromCenter;
	}

	public void setAveragePointDistanceFromCenter(double averageDistanceFromCenter) {
		this.averagePointDistanceFromCenter = averageDistanceFromCenter;
	}

	public double getTotalPointDistanceFromCenter() {
		return totalPointDistanceFromCenter;
	}

	public void setTotalPointDistanceFromCenter(double sumOfSquaredError) {
		this.totalPointDistanceFromCenter = sumOfSquaredError;
	}

	public Map<String, Double> getPointDistancesFromCenter() {
		return pointDistancesFromCenter;
	}

	public void setPointDistancesFromCenter(Map<String, Double> distancesFromCenter) {
		this.pointDistancesFromCenter = distancesFromCenter;
	}

}
