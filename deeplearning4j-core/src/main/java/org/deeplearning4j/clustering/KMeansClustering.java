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

package org.deeplearning4j.clustering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.distancefunction.DistanceFunction;
import org.nd4j.linalg.distancefunction.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * adapted to ndarray matrices
 * 
 * @author Adam Gibson
 *
 */
public class KMeansClustering implements ClusteringAlgorithm, Serializable {

	private static final long					serialVersionUID	= 338231277453149972L;
	private static final Logger						log					= LoggerFactory.getLogger(KMeansClustering.class);
	private static int							defaultIterationCount = 100;

	private Class<? extends DistanceFunction>	distanceFunction;
	private int									iterationCount;
	private transient ExecutorService			exec;
	
	private Integer								nbCluster;
	private ClusterSet							clusterSet;

	public KMeansClustering(Integer nbCluster, int iterationCount, Class<? extends DistanceFunction> distanceFunction) {
		this.nbCluster = nbCluster;
		this.iterationCount = iterationCount;
		this.distanceFunction = distanceFunction;
	}

	public KMeansClustering(Integer nbCluster) {
		this(nbCluster, defaultIterationCount, EuclideanDistance.class);
	}

	public ClusterSet applyTo(List<INDArray> points) {
		exec = Executors.newScheduledThreadPool(Runtime.getRuntime().availableProcessors());
		initClusters(points);
		iterations(points);
		return clusterSet;
	}

	private void iterations(List<INDArray> points) {
		for( int i=0;i<iterationCount;i++) {
			clusterSet.addPoints(points, true);
			clusterSet.removePoints();
		}
	}

	
	protected void initClusters(List<INDArray> initialPointsList) {
		
		List<INDArray> points = new ArrayList<INDArray>(initialPointsList);
		
		clusterSet = new ClusterSet(distanceFunction);
		Random random = new Random();
		final INDArray center = points.remove(random.nextInt(points.size())).linearView();
		clusterSet.addNewClusterWithCenter(center);
		
		while( clusterSet.getClusterCount()<nbCluster ) {
			INDArray dxs = computeDxs(points);
			double r = random.nextFloat() * dxs.getDouble(dxs.length() - 1);
			for (int i = 0; i < dxs.length(); i++) {
				if (dxs.getDouble(i) >= r) {
					clusterSet.addNewClusterWithCenter(points.remove(i));
					break;
				}
			}
		}
			
	}

	protected INDArray computeDxs(final List<INDArray> points) {
		final INDArray dxs = Nd4j.create(points.size(), points.get(0).columns());

		final CountDownLatch latch = new CountDownLatch(points.size());
		for (int i = 0; i < points.size(); i++) {
			final int i2 = i;
			exec.execute(new Runnable() {
				@Override
				public void run() {
					INDArray point = points.get(i2);
					dxs.putScalar(i2, (int) Math.pow(clusterSet.getDistanceFromNearestCluster(point), 2));
					latch.countDown();
				}
			});

		}

		try {
			latch.await();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		}

		return dxs;
	}


}
