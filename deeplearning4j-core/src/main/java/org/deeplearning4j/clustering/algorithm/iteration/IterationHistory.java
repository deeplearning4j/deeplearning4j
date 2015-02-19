package org.deeplearning4j.clustering.algorithm.iteration;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.clustering.cluster.info.ClusterSetInfo;

public class IterationHistory {

	private Map<Integer, IterationInfo> iterationsInfos = new HashMap<Integer, IterationInfo>();
	
	public ClusterSetInfo getMostRecentClusterSetInfo() {
		IterationInfo iterationInfo = getMostRecentIterationInfo();
		return iterationInfo==null ? null : iterationInfo.getClusterSetInfo();
	}
	public IterationInfo getMostRecentIterationInfo() {
		return getIterationInfo(getIterationCount()-1);
	}
	public int getIterationCount() {
		return getIterationsInfos().size();
	}
	public IterationInfo getIterationInfo(int iterationIdx) {
		return getIterationsInfos().get(iterationIdx);
	}

	public Map<Integer, IterationInfo> getIterationsInfos() {
		return iterationsInfos;
	}

	public void setIterationsInfos(Map<Integer, IterationInfo> iterationsInfos) {
		this.iterationsInfos = iterationsInfos;
	}
	
	
}
