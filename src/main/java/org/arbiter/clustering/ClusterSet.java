/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.clustering;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;

public class ClusterSet {

	private Class<? extends Accumulation>	distanceFunction;
	private List<Cluster>						clusters	= new ArrayList<>();

	public ClusterSet() {

	}

	public ClusterSet(INDArray centers) {
		for (Integer idx = 0, count = centers.rows(); idx < count; idx++) {
			clusters.add(new Cluster(centers.getRow(idx)));
		}
	}
	
	public ClusterSet(Class<? extends Accumulation> distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

	
	public void addNewClusterWithCenter(INDArray center) {
		getClusters().add(new Cluster(center));
	}
	
	public INDArray getCenters() {
		INDArray centers = Nd4j.create(clusters.size(), clusters.get(0).getCenter().columns());
		for (Integer idx = 0, count = clusters.size(); idx < count; idx++) {
			centers.putRow(idx, clusters.get(idx).getCenter());
		}
		return centers;
	}
	
	public void addPoint(INDArray point) {
		nearestCluster(point).addPoint(point, true);
	}
	public void addPoint(INDArray point, boolean moveClusterCenter) {
		nearestCluster(point).addPoint(point, moveClusterCenter);
	}
	
	public void addPoints(List<INDArray> points) {
		addPoints(points, true);
	}
	public void addPoints(List<INDArray> points, boolean moveClusterCenter) {
		for( INDArray point : points )
			addPoint(point, moveClusterCenter);
	}
	
	public Cluster classify(INDArray point) {
		return classify(point, distanceFunction);
	}

	public Cluster classify(INDArray point, Class<? extends Accumulation> distanceFunction) {
		return nearestCluster(point);
	}

	protected Cluster nearestCluster(INDArray point) {

		Cluster nearestCluster = null;
		double minDistance = Float.MAX_VALUE;

		double currentDistance;
		for (Cluster cluster : getClusters()) {
			INDArray currentCenter = cluster.getCenter();
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

	private double getDistance(INDArray m1, INDArray m2) {
      return Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(m1,m2)).currentResult().doubleValue();
	}
	
	public double getDistanceFromNearestCluster(INDArray point) {
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

	public Class<? extends Accumulation> getDistanceFunction() {
		return distanceFunction;
	}

	public void setDistanceFunction(Class<? extends Accumulation> distanceFunction) {
		this.distanceFunction = distanceFunction;
	}

}
