/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.clustering.cluster;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

public class ClusterSet implements Serializable {

	private String	distanceFunction;
	private List<Cluster> clusters;
	private Map<String, String>	pointDistribution;

	public ClusterSet() {
		this(null);
	}

	public ClusterSet(String distanceFunction) {
		this.distanceFunction = distanceFunction;
		this.clusters = Collections.synchronizedList(new ArrayList<Cluster>());
		this.pointDistribution = Collections.synchronizedMap(new HashMap<String, String>());
	}
	
	public Cluster addNewClusterWithCenter(Point center) {
		Cluster newCluster = new Cluster(center, distanceFunction);
		getClusters().add(newCluster);
		setPointLocation(center, newCluster);
		return newCluster;
	}

	public PointClassification classifyPoint(Point point) {
		return classifyPoint(point, true);
	}
	
	public void classifyPoints(List<Point> points) {
		classifyPoints(points, true);
	}

	public void classifyPoints(List<Point> points, boolean moveClusterCenter) {
		for (Point point : points)
			classifyPoint(point, moveClusterCenter);
	}

	public PointClassification classifyPoint(Point point, boolean moveClusterCenter) {
		Pair<Cluster, Double> nearestCluster = nearestCluster(point);
		Cluster newCluster = nearestCluster.getFirst();
		boolean locationChange = isPointLocationChange(point, newCluster);
		addPointToCluster(point, newCluster, moveClusterCenter);
		return new PointClassification(nearestCluster.getFirst(), nearestCluster.getSecond(), locationChange);
	}
	
	private boolean isPointLocationChange(Point point, Cluster newCluster) {
		if( !getPointDistribution().containsKey(point.getId()) )
			return true;
		return !getPointDistribution().get(point.getId()).equals(newCluster.getId());
	}
	
	private void addPointToCluster(Point point, Cluster cluster, boolean moveClusterCenter) {
		cluster.addPoint(point, moveClusterCenter);
		setPointLocation(point, cluster);
	}
	
	private void setPointLocation(Point point, Cluster cluster) {
		pointDistribution.put(point.getId(), cluster.getId());
	}

	
	public Pair<Cluster, Double> nearestCluster(Point point) {

		Cluster nearestCluster = null;
		double minDistance = Float.MAX_VALUE;

		double currentDistance;
		for (Cluster cluster : getClusters()) {
			currentDistance = cluster.getDistanceToCenter(point);
			if (currentDistance < minDistance) {
				minDistance = currentDistance;
				nearestCluster = cluster;
			}
		}

		return new Pair<Cluster, Double>(nearestCluster, minDistance);

	}

	public double getDistance(Point m1, Point m2) {
		return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createAccum(distanceFunction,m1.getArray(),m2.getArray())).currentResult().doubleValue();
    }

	public double getDistanceFromNearestCluster(Point point) {
		return nearestCluster(point).getSecond();
	}

	public String getClusterCenterId(String clusterId) {
		Point clusterCenter = getClusterCenter(clusterId);
		return clusterCenter == null ? null : clusterCenter.getId();
	}

	public Point getClusterCenter(String clusterId) {
		Cluster cluster = getCluster(clusterId);
		return cluster == null ? null : cluster.getCenter();
	}

	public Cluster getCluster(String id) {
		for (int i = 0, j = clusters.size(); i < j; i++)
			if (id.equals(clusters.get(i).getId()))
				return clusters.get(i);
		return null;
	}

	public int getClusterCount() {
		return getClusters() == null ? 0 : getClusters().size();
	}

	public void removePoints() {
		for (Cluster cluster : getClusters())
			cluster.removePoints();
	}

	public List<Cluster> getMostPopulatedClusters(int count) {
		List<Cluster> mostPopulated = new ArrayList<Cluster>(clusters);
		Collections.sort(mostPopulated, new Comparator<Cluster>() {
			public int compare(Cluster o1, Cluster o2) {
				return new Integer(o1.getPoints().size()).compareTo(new Integer(o2.getPoints().size()));
			}
		});
		return mostPopulated.subList(0, count);
	}

	public List<Cluster> removeEmptyClusters() {
		List<Cluster> emptyClusters = new ArrayList<Cluster>();
		for (Cluster cluster : clusters)
			if (cluster.isEmpty())
				emptyClusters.add(cluster);
		clusters.removeAll(emptyClusters);
		return emptyClusters;
	}

	public List<Cluster> getClusters() {
		return clusters;
	}

	public void setClusters(List<Cluster> clusters) {
		this.clusters = clusters;
	}

	public String getAccumulation() {
		return distanceFunction;
	}

	public void setAccumulation(String distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

	public Map<String, String> getPointDistribution() {
		return pointDistribution;
	}

	public void setPointDistribution(Map<String, String> pointDistribution) {
		this.pointDistribution = pointDistribution;
	}

}
