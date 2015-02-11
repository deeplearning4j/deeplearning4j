package org.deeplearning4j.clustering.cluster;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class Cluster {

	private String			id = UUID.randomUUID().toString();
	
	private Point		center;
	private List<Point>	points = new ArrayList<Point>();

	public Cluster() {
		super();
	}
	
	public Cluster(String id) {
		super();
		this.id = id;
	}

	public Cluster(Point center) {
		super();
		this.center = center;
	}
	
	public Cluster(String id, Point center) {
		super();
		this.id = id;
		this.center = center;
	}

	public Cluster(Point center, List<Point> points) {
		super();
		this.center = center;
		this.points = points;
	}
	
	public void addPoint(Point point) {
		addPoint(point, true);
	}
	public void addPoint(Point point, boolean moveClusterCenter) {
		if( moveClusterCenter ) {
			center.muli(points.size()).addi(point).divi(points.size()+1);
		}
		getPoints().add(point);
	}
	
	public void removePoints() {
		if( getPoints()!=null )
			getPoints().clear();
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
	

}
