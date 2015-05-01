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

package org.deeplearning4j.clustering;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Cluster {

	private String			id = UUID.randomUUID().toString();
	
	private INDArray		center;
	private List<INDArray>	points = new ArrayList<INDArray>();

	public Cluster() {
		super();
	}
	
	public Cluster(String id) {
		super();
		this.id = id;
	}

	public Cluster(INDArray center) {
		super();
		this.center = center;
	}
	
	public Cluster(String id, INDArray center) {
		super();
		this.id = id;
		this.center = center;
	}

	public Cluster(INDArray center, List<INDArray> points) {
		super();
		this.center = center;
		this.points = points;
	}
	
	public void addPoint(INDArray point) {
		addPoint(point, true);
	}
	public void addPoint(INDArray point, boolean moveClusterCenter) {
		if( moveClusterCenter ) {
			center.muli(points.size()).addi(point).divi(points.size()+1);
		}
		getPoints().add(point);
	}
	
	public void removePoints() {
		if( getPoints()!=null )
			getPoints().clear();
	}

	public INDArray getCenter() {
		return center;
	}

	public void setCenter(INDArray center) {
		this.center = center;
	}

	public List<INDArray> getPoints() {
		return points;
	}

	public void setPoints(List<INDArray> points) {
		this.points = points;
	}

	

}
