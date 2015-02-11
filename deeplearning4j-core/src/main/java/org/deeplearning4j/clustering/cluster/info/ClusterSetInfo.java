package org.deeplearning4j.clustering.cluster.info;

import java.util.HashMap;
import java.util.Map;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

public class ClusterSetInfo {

	private Map<String, ClusterInfo>		clustersInfos					= new HashMap<String, ClusterInfo>();
	private Table<String, String, Double>	distancesBetweenClustersCenters	= HashBasedTable.create();

	
	public double getAverageClusterPointsDistanceFromOwnCenter() {
		if( clustersInfos.size()==0 )
			return 0;
		
		double average = 0;
		for( int i=0,j=clustersInfos.size();i<j;i++ )
			average += clustersInfos.get(i).getAveragePointDistanceFromCenter();
		return average / clustersInfos.size();
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

}
