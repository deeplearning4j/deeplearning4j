package org.deeplearning4j.clustering.cluster;

import org.deeplearning4j.clustering.cluster.info.ClusterInfo;
import org.deeplearning4j.clustering.cluster.info.ClusterSetInfo;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.DistanceFunction;

public class ClusterUtils {

	public static ClusterSetInfo computeClusterSetInfo(ClusterSet clusterSet) {
		ClusterSetInfo info = new ClusterSetInfo();
		int clusterCount = clusterSet.getClusterCount();
		
		for(int i=0,j=clusterCount;i<j;i++) {
			Cluster cluster = clusterSet.getClusters().get(i);
			info.getClustersInfos().put(cluster.getId(), computeClusterInfos(cluster, clusterSet.getDistanceFunction()));
		}
		
		for(int i=0,j=clusterCount;i<j;i++) {
			Cluster fromCluster = clusterSet.getClusters().get(i);
			DistanceFunction fromClusterDistanceFunction = distanceFunction(clusterSet.getDistanceFunction(), (INDArray) fromCluster.getCenter());
			for(int k=i+1,l=clusterSet.getClusterCount();k<l;k++) {
				Cluster toCluster = clusterSet.getClusters().get(k);
				double distance = fromClusterDistanceFunction.apply((INDArray) toCluster.getCenter());
				info.getDistancesBetweenClustersCenters().put(fromCluster.getId(), toCluster.getId(), distance);
			}
		}
		
		return info;
	}
	
	public static ClusterInfo computeClusterInfos(Cluster cluster, Class<? extends DistanceFunction> distanceFunctionClass) {
		ClusterInfo info = new ClusterInfo();
		DistanceFunction centerDistanceFunction = distanceFunction(distanceFunctionClass, (INDArray) cluster.getCenter());
		for(int i=0,j=cluster.getPoints().size();i<j;i++) {
			Point point = cluster.getPoints().get(i);
			double distance = centerDistanceFunction.apply(point);
			info.setAveragePointDistanceFromCenter(info.getAveragePointDistanceFromCenter()+distance);
			info.getPointDistancesFromCenter().put(point.getId(), distance);
		}
		info.setTotalPointDistanceFromCenter(info.getAveragePointDistanceFromCenter());
		if( cluster.getPoints().size()>0 )
			info.setAveragePointDistanceFromCenter(info.getAveragePointDistanceFromCenter()/cluster.getPoints().size());
		return info;
	}
	
	public static DistanceFunction distanceFunction(Class<? extends DistanceFunction> distanceFunction, INDArray m1) {
		try {
			return distanceFunction.getConstructor(INDArray.class).newInstance(m1);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
}
