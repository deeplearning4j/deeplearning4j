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

package org.arbiter.clustering.algorithm.optimisation;

public class ClusteringOptimization {

	private ClusteringOptimizationType	type;
	private double						value;

	

	public ClusteringOptimization(ClusteringOptimizationType type, double value) {
		super();
		this.type = type;
		this.value = value;
	}

	public ClusteringOptimizationType getType() {
		return type;
	}

	public void setType(ClusteringOptimizationType type) {
		this.type = type;
	}

	public double getValue() {
		return value;
	}

	public void setValue(double value) {
		this.value = value;
	}

}
