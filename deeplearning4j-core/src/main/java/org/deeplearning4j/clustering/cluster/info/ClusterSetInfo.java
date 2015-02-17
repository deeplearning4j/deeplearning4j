package org.deeplearning4j.clustering.cluster.info;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

public class ClusterSetInfo {

	private Map<String, ClusterInfo>		clustersInfos					= new HashMap<String, ClusterInfo>();
	private Table<String, String, Double>	distancesBetweenClustersCenters	= HashBasedTable.create();
	private AtomicInteger					pointLocationChange;
	private boolean							threadSafe;

	public ClusterSetInfo() {
		this(false);
	}

	public ClusterSetInfo(boolean threadSafe) {
		this.pointLocationChange = new AtomicInteger(0);
		this.threadSafe = threadSafe;
		if (threadSafe) {
			clustersInfos = Collections.synchronizedMap(clustersInfos);
		}
	}
	
	public static ClusterSetInfo initialize(ClusterSet clusterSet, boolean threadSafe) {
		ClusterSetInfo info = new ClusterSetInfo();
		for(int i=0,j=clusterSet.getClusterCount();i<j;i++)
			info.addClusterInfo(clusterSet.getClusters().get(i).getId());
		return info;
	}
	
	public void removeClusterInfos(List<Cluster> clusters) {
		for( Cluster cluster : clusters ) {
			clustersInfos.remove(cluster.getId());
		}
	}
	
	public ClusterInfo addClusterInfo(String clusterId) {
		ClusterInfo clusterInfo = new ClusterInfo(threadSafe);
		clustersInfos.put(clusterId, clusterInfo);
		return clusterInfo;
	}

	public ClusterInfo getClusterInfo(String clusterId) {
		return clustersInfos.get(clusterId);
	}

	public double getAveragePointDistanceFromClusterCenter() {
		if (clustersInfos == null || clustersInfos.size() == 0)
			return 0;

		double average = 0;
		for (ClusterInfo info : clustersInfos.values())
			average += info.getAveragePointDistanceFromCenter();
		return average / clustersInfos.size();
	}
	
	public double getPointDistanceFromClusterVariance() {
		if (clustersInfos == null || clustersInfos.size() == 0)
			return 0;

		double average = 0;
		for (ClusterInfo info : clustersInfos.values())
			average += info.getPointDistanceFromCenterVariance();
		return average / clustersInfos.size();
	}
	
	public int getPointsCount() {
		int count = 0;
		for(ClusterInfo clusterInfo : clustersInfos.values())
			count += clusterInfo.getPointDistancesFromCenter().size();
		return count;
	}
	
	

	public Map<String, ClusterInfo> getClustersInfos() {
		return clustersInfos;
	}

	public void setClustersInfos(Map<String, ClusterInfo> clustersInfos) {
		this.clustersInfos = clustersInfos;
	}

	public Table<String, String, Double> getDistancesBetweenClustersCenters() {
		return distancesBetweenClustersCenters;
	}

	public void setDistancesBetweenClustersCenters(Table<String, String, Double> interClusterDistances) {
		this.distancesBetweenClustersCenters = interClusterDistances;
	}

	public AtomicInteger getPointLocationChange() {
		return pointLocationChange;
	}

	public void setPointLocationChange(AtomicInteger pointLocationChange) {
		this.pointLocationChange = pointLocationChange;
	}

}
