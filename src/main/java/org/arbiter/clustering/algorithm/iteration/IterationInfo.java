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

package org.arbiter.clustering.algorithm.iteration;

import org.arbiter.clustering.cluster.info.ClusterSetInfo;

public class IterationInfo {

	private int							index;
	private ClusterSetInfo clusterSetInfo;
	private boolean						strategyApplied;
	
	
	public IterationInfo(int index) {
		super();
		this.index = index;
	}
	
	public IterationInfo(int index, ClusterSetInfo clusterSetInfo) {
		super();
		this.index = index;
		this.clusterSetInfo = clusterSetInfo;
	}
	
	

	public int getIndex() {
		return index;
	}
	public void setIndex(int index) {
		this.index = index;
	}
	public ClusterSetInfo getClusterSetInfo() {
		return clusterSetInfo;
	}
	public void setClusterSetInfo(ClusterSetInfo clusterSetInfo) {
		this.clusterSetInfo = clusterSetInfo;
	}

	public boolean isStrategyApplied() {
		return strategyApplied;
	}

	public void setStrategyApplied(boolean optimizationApplied) {
		this.strategyApplied = optimizationApplied;
	}

	
}
