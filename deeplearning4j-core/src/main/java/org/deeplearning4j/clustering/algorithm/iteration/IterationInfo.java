package org.deeplearning4j.clustering.algorithm.iteration;

import org.deeplearning4j.clustering.cluster.info.ClusterSetInfo;

public class IterationInfo {

	private int							index;
	private ClusterSetInfo				clusterSetInfo;
	
	
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

	
}
