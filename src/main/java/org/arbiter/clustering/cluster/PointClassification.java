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

package org.arbiter.clustering.cluster;

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
