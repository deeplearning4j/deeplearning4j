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

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.nd4j.linalg.factory.Nd4j;

public class Cluster {

	private String id = UUID.randomUUID().toString();
	private String label;

	private Point center;
	private List<Point>	points	= new ArrayList<>();

	private String distanceFunction;

	public Cluster() {
		super();
	}

	public Cluster(Point center,String distanceFunction) {
		this.distanceFunction = distanceFunction;
		setCenter(center);
	}

	


	public double getDistanceToCenter(Point point) {
		return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createAccum(distanceFunction,center.getArray(),point.getArray())).currentResult().doubleValue();
    }

	public void addPoint(Point point) {
		addPoint(point, true);
	}

	public void addPoint(Point point, boolean moveClusterCenter) {
		if (moveClusterCenter) {
			center.muli(points.size()).addi(point).divi(points.size() + 1);
		}
		getPoints().add(point);
	}

	public void removePoints() {
		if (getPoints() != null)
			getPoints().clear();
	}

	public boolean isEmpty() {
		return points == null || points.size() == 0;
	}

	public Point getPoint(String id) {
		for (Point point : points)
			if (id.equals(point.getId()))
				return point;
		return null;
	}

	public Point removePoint(String id) {
		Point removePoint = null;
		for (Point point : points)
			if (id.equals(point.getId()))
				removePoint = point;
		if (removePoint != null)
			points.remove(removePoint);
		return removePoint;
	}

	public Point getCenter() {
		return center;
	}

	public void setCenter(Point center) {
		this.center = center;
	}

	public List<Point> getPoints() {
		return points;
	}

	public void setPoints(List<Point> points) {
		this.points = points;
	}

	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

}
