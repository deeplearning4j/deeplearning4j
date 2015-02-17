package org.deeplearning4j.clustering.cluster;

public class PointClassification {

	private Cluster	cluster;
	private double	distanceFromCenter;
	private boolean	newLocation;

	public PointClassification(Cluster cluster, double distanceFromCenter, boolean newLocation) {
		super();
		this.cluster = cluster;
		this.distanceFromCenter = distanceFromCenter;
		this.newLocation = newLocation;
	}

	public Cluster getCluster() {
		return cluster;
	}

	public void setCluster(Cluster cluster) {
		this.cluster = cluster;
	}

	public double getDistanceFromCenter() {
		return distanceFromCenter;
	}

	public void setDistanceFromCenter(double distanceFromCenter) {
		this.distanceFromCenter = distanceFromCenter;
	}

	public boolean isNewLocation() {
		return newLocation;
	}

	public void setNewLocation(boolean newLocation) {
		this.newLocation = newLocation;
	}

}
