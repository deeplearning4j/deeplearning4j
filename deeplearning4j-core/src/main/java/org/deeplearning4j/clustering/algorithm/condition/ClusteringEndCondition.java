package org.deeplearning4j.clustering.algorithm.condition;

import org.deeplearning4j.clustering.algorithm.iteration.IterationHistory;
import org.deeplearning4j.clustering.algorithm.iteration.IterationInfo;
import org.deeplearning4j.clustering.cluster.info.ClusterInfo;
import org.deeplearning4j.clustering.cluster.info.ClusterSetInfo;
import org.nd4j.linalg.indexing.conditions.Condition;
import org.nd4j.linalg.indexing.conditions.LessThan;

public class ClusteringEndCondition {

	private Condition	iterationCountCondition;
	private Condition	averagePointDistanceToClusterCenterCondition;
	private Condition	averagePointDistanceToClusterCenterProgressionRateCondition;
	
	public ClusteringEndCondition iterationCountLessThan(int iterationCount) {
		ClusteringEndCondition condition = new ClusteringEndCondition();
		condition.iterationCountCondition = new LessThan(iterationCount);
		return condition;
	}
	
	public ClusteringEndCondition averagePointDistanceToClusterCenterLessThan(double averagePointDistanceToClusterCenter) {
		ClusteringEndCondition condition = new ClusteringEndCondition();
		condition.averagePointDistanceToClusterCenterCondition = new LessThan(averagePointDistanceToClusterCenter);
		return condition;
	}
	
	public ClusteringEndCondition averagePointDistanceToClusterCenterProgressionRateConditionLessThan(double averagePointDistanceToClusterCenterProgressionRateCondition) {
		ClusteringEndCondition condition = new ClusteringEndCondition();
		condition.averagePointDistanceToClusterCenterProgressionRateCondition = new LessThan(averagePointDistanceToClusterCenterProgressionRateCondition);
		return condition;
	}
	
	
	public boolean isSatisfied(IterationHistory iterationHistory) {
		boolean satisfied = true;
		if( isIterationCountConditionDefined() )
			satisfied &= isIterationCountConditionSatisfied(iterationHistory==null?0:iterationHistory.getIterationCount());
		
		ClusterSetInfo clusterSetInfo = iterationHistory.getMostRecentClusterSetInfo();
		if( clusterSetInfo==null )
			return false;
		
		if( isAveragePointDistanceToClusterCenterConditionDefined() )
			satisfied &= isAveragePointDistanceToClusterCenterConditionSatisfied(clusterSetInfo);
		if( isAveragePointDistanceToClusterCenterProgressionRateConditionDefined() )
			satisfied &= isAveragePointDistanceToClusterCenterProgressionRateConditionSatisfied(iterationHistory);
		
		return satisfied;
	}
	
	protected boolean isIterationCountConditionDefined() {
		return iterationCountCondition!=null;
	}
	protected boolean isIterationCountConditionSatisfied(int currentIterationCount) {
		return iterationCountCondition.apply(currentIterationCount);
	}
	
	
	
	protected boolean isAveragePointDistanceToClusterCenterConditionDefined() {
		return averagePointDistanceToClusterCenterCondition!=null;
	}
	protected boolean isAveragePointDistanceToClusterCenterConditionSatisfied(ClusterSetInfo clusterSetInfo) {
		for(int i=0,j=clusterSetInfo.getClustersInfos().size();i<j;i++) {
			ClusterInfo clusterInfo = clusterSetInfo.getClustersInfos().get(i);
			if( !averagePointDistanceToClusterCenterCondition.apply(clusterInfo.getAveragePointDistanceFromCenter()) )
				return false;
		}
		return true;
	}
	
	protected boolean isAveragePointDistanceToClusterCenterProgressionRateConditionDefined() {
		return averagePointDistanceToClusterCenterProgressionRateCondition!=null;
	}
	protected boolean isAveragePointDistanceToClusterCenterProgressionRateConditionSatisfied(IterationHistory iterationHistory) {
		if(iterationHistory.getIterationCount()<=1)
			return false;
		
		IterationInfo referenceIteration = iterationHistory.getIterationInfo(iterationHistory.getIterationCount()-1);
		IterationInfo iteration = iterationHistory.getIterationInfo(iterationHistory.getIterationCount());
		
		double variation = iteration.getClusterSetInfo().getAverageClusterPointsDistanceFromOwnCenter()-referenceIteration.getClusterSetInfo().getAverageClusterPointsDistanceFromOwnCenter();
		variation /= referenceIteration.getClusterSetInfo().getAverageClusterPointsDistanceFromOwnCenter();
		
		return averagePointDistanceToClusterCenterProgressionRateCondition.apply(variation);
	}
	
	
	
	
	public Condition getIterationCountCondition() {
		return iterationCountCondition;
	}
	public void setIterationCountCondition(Condition iterationCount) {
		this.iterationCountCondition = iterationCount;
	}
	public Condition getAveragePointDistanceToClusterCenterCondition() {
		return averagePointDistanceToClusterCenterCondition;
	}
	public void setAveragePointDistanceToClusterCenterCondition(Condition averagePointDistanceToClusterCenter) {
		this.averagePointDistanceToClusterCenterCondition = averagePointDistanceToClusterCenter;
	}
	public Condition getAveragePointDistanceToClusterCenterProgressionRateCondition() {
		return averagePointDistanceToClusterCenterProgressionRateCondition;
	}
	public void setAveragePointDistanceToClusterCenterProgressionRateCondition(Condition averagePointDistanceToClusterCenterProgressionRate) {
		this.averagePointDistanceToClusterCenterProgressionRateCondition = averagePointDistanceToClusterCenterProgressionRate;
	}
	
	
}
