package org.deeplearning4j.clustering.cluster;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public class Cluster {

	private String								id		= UUID.randomUUID().toString();
	private String								label;

	private Point								center;
	private List<Point>							points	= new ArrayList<Point>();

	private Class<? extends DistanceFunction>	distanceFunctionClass;
	private DistanceFunction					distanceToCenterFunction;

	public Cluster() {
		super();
	}

	public Cluster(Point center, Class<? extends DistanceFunction> distanceFunctionClass) {
		this.distanceFunctionClass = distanceFunctionClass;
		setCenter(center);
	}

	

	private void refreshDistanceToCurrentCenterFunction() {
		try {
			distanceToCenterFunction = distanceFunctionClass.getConstructor(INDArray.class).newInstance(center.getArray());
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public double getDistanceToCenter(Point point) {
		return distanceToCenterFunction.apply(point);
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
		refreshDistanceToCurrentCenterFunction();
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
