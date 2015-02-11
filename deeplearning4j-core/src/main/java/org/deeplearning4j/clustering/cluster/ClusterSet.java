package org.deeplearning4j.clustering.cluster;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.distancefunction.DistanceFunction;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

public class ClusterSet {

	private Class<? extends DistanceFunction>	distanceFunction;
	private List<Cluster>						clusters	= new ArrayList<Cluster>();
	
	public ClusterSet() {

	}

	public ClusterSet(Class<? extends DistanceFunction> distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

	
	public void addNewClusterWithCenter(Point center) {
		getClusters().add(new Cluster(center));
	}
	
	
	
	public void addPoint(Point point) {
		nearestCluster(point).addPoint(point, true);
	}
	public void addPoint(Point point, boolean moveClusterCenter) {
		nearestCluster(point).addPoint(point, moveClusterCenter);
	}
	
	public void addPoints(List<Point> points) {
		addPoints(points, true);
	}
	public void addPoints(List<Point> points, boolean moveClusterCenter) {
		for( Point point : points )
			addPoint(point, moveClusterCenter);
	}
	
	public Cluster classify(Point point) {
		return classify(point, distanceFunction);
	}

	public Cluster classify(Point point, Class<? extends DistanceFunction> distanceFunction) {
		return nearestCluster(point);
	}

	protected Cluster nearestCluster(Point point) {

		Cluster nearestCluster = null;
		double minDistance = Float.MAX_VALUE;

		double currentDistance;
		for (Cluster cluster : getClusters()) {
			Point currentCenter = cluster.getCenter();
			if (currentCenter != null) {
				currentDistance = getDistance(currentCenter, point);
				if (currentDistance < minDistance) {
					minDistance = currentDistance;
					nearestCluster = cluster;
				}
			}
		}

		return nearestCluster;
	}

	private double getDistance(Point m1, Point m2) {
		DistanceFunction function;
		try {
			function = distanceFunction.getConstructor(Point.class).newInstance(m1);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
		return function.apply(m2);
	}
	
	public double getDistanceFromNearestCluster(Point point) {
		Cluster nearestCluster = nearestCluster(point);
		return getDistance(nearestCluster.getCenter(), point);
	}
	
	public int getClusterCount() {
		return getClusters()==null ? 0 : getClusters().size();
	}
	
	public void removePoints() {
		for(Cluster cluster : getClusters() )
			cluster.removePoints();
	}

	public List<Cluster> getClusters() {
		return clusters;
	}

	public void setClusters(List<Cluster> clusters) {
		this.clusters = clusters;
	}

	public Class<? extends DistanceFunction> getDistanceFunction() {
		return distanceFunction;
	}

	public void setDistanceFunction(Class<? extends DistanceFunction> distanceFunction) {
		this.distanceFunction = distanceFunction;
	}
	
	

}
