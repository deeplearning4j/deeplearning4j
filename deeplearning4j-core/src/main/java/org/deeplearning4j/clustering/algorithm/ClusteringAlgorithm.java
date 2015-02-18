package org.deeplearning4j.clustering.algorithm;

import java.util.List;

import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;

public interface ClusteringAlgorithm {

	ClusterSet applyTo(List<Point> points);
	
}
