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

package org.deeplearning4j.clustering.cluster.info;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

public class ClusterInfo {

	private double averagePointDistanceFromCenter;
	private double maxPointDistanceFromCenter;
	private double pointDistanceFromCenterVariance;
	private double totalPointDistanceFromCenter;
	private Map<String, Double>	pointDistancesFromCenter = new ConcurrentHashMap<>();

	public ClusterInfo() {
		this(false);
	}

	public ClusterInfo(boolean threadSafe) {
		super();
		if (threadSafe) {
			pointDistancesFromCenter = Collections.synchronizedMap(pointDistancesFromCenter);
		}
	}

	public Set<Map.Entry<String, Double>> getSortedPointDistancesFromCenter() {
		SortedSet<Map.Entry<String, Double>> sortedEntries = new TreeSet<>(new Comparator<Map.Entry<String, Double>>() {
			@Override
			public int compare(Map.Entry<String, Double> e1, Map.Entry<String, Double> e2) {
				int res = e1.getValue().compareTo(e2.getValue());
				return res != 0 ? res : 1;
			}
		});
		sortedEntries.addAll(pointDistancesFromCenter.entrySet());
		return sortedEntries;
	}

	public Set<Map.Entry<String, Double>> getReverseSortedPointDistancesFromCenter() {
		SortedSet<Map.Entry<String, Double>> sortedEntries = new TreeSet<>(new Comparator<Map.Entry<String, Double>>() {
			@Override
			public int compare(Map.Entry<String, Double> e1, Map.Entry<String, Double> e2) {
				int res = e1.getValue().compareTo(e2.getValue());
				return -(res != 0 ? res : 1);
			}
		});
		sortedEntries.addAll(pointDistancesFromCenter.entrySet());
		return sortedEntries;
	}

	public List<String> getPointsFartherFromCenterThan(double maxDistance) {
		Set<Map.Entry<String, Double>> sorted = getReverseSortedPointDistancesFromCenter();
		List<String> ids = new ArrayList<>();
		for (Map.Entry<String, Double> entry : sorted) {
			if (entry.getValue() < maxDistance)
				break;
			ids.add(entry.getKey());
		}
		return ids;
	}

	
	public double getAveragePointDistanceFromCenter() {
		return averagePointDistanceFromCenter;
	}

	public void setAveragePointDistanceFromCenter(double averageDistanceFromCenter) {
		this.averagePointDistanceFromCenter = averageDistanceFromCenter;
	}

	public double getTotalPointDistanceFromCenter() {
		return totalPointDistanceFromCenter;
	}

	public void setTotalPointDistanceFromCenter(double sumOfSquaredError) {
		this.totalPointDistanceFromCenter = sumOfSquaredError;
	}

	public Map<String, Double> getPointDistancesFromCenter() {
		return pointDistancesFromCenter;
	}

	public void setPointDistancesFromCenter(Map<String, Double> distancesFromCenter) {
		this.pointDistancesFromCenter = distancesFromCenter;
	}

	public double getPointDistanceFromCenterVariance() {
		return pointDistanceFromCenterVariance;
	}

	public void setPointDistanceFromCenterVariance(double pointDistanceFromCenterVariance) {
		this.pointDistanceFromCenterVariance = pointDistanceFromCenterVariance;
	}

	public double getMaxPointDistanceFromCenter() {
		return maxPointDistanceFromCenter;
	}

	public void setMaxPointDistanceFromCenter(double maxPointDistanceFromCenter) {
		this.maxPointDistanceFromCenter = maxPointDistanceFromCenter;
	}

}
