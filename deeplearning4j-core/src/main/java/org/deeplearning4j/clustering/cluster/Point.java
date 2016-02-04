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
import java.util.List;
import java.util.UUID;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Point  implements Serializable {

	private static final long	serialVersionUID = -6658028541426027226L;

	private String id = UUID.randomUUID().toString();
	private String label;
	private INDArray array;

	public Point(INDArray array) {
		super();
		this.array = array;
	}

	public Point(String id, INDArray array) {
		super();
		this.id = id;
		this.array = array;
	}

	public Point(String id, String label, double[] data) {
		this(id, label, Nd4j.create(data));
	}
	
	public Point(String id, String label, INDArray array) {
		super();
		this.id = id;
		this.label = label;
		this.array = array;
	}

	public static List<Point> toPoints(List<INDArray> vectors) {
		List<Point> points = new ArrayList<>();
		for (INDArray vector : vectors)
			points.add(new Point(vector));
		return points;
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

	public INDArray getArray() {
		return array;
	}

	public void setArray(INDArray array) {
		this.array = array;
	}


}
