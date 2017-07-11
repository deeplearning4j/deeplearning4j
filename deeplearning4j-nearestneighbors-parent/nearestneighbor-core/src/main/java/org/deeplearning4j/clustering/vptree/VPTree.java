/*-
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

package org.deeplearning4j.clustering.vptree;

import lombok.*;
import org.deeplearning4j.clustering.berkeley.PriorityQueue;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.sptree.HeapItem;
import org.deeplearning4j.clustering.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Dot;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Vantage point tree implementation
 *
 * @author Adam Gibson
 */
@Builder
@AllArgsConstructor
public class VPTree {

    public static final String EUCLIDEAN = "euclidean";
    private double tau;
    @Getter
    @Setter
    private INDArray items;
    private Node root;
    private String similarityFunction;
    @Getter
    private boolean invert = false;
    private ExecutorService executorService;
    @Getter
    private boolean parallel = true;
    private AtomicInteger size = new AtomicInteger(0);

    /**
     *
     * @param points
     * @param invert
     */
    public VPTree(INDArray points,boolean invert) {
        this(points,"euclidean",true,invert);
    }

    /**
     *
     * @param points
     * @param invert
     * @param parallel
     */
    public VPTree(INDArray points,boolean invert,boolean parallel) {
        this(points,"euclidean",parallel,invert);
    }

    /**
     *
     * @param items the items to use
     * @param similarityFunction the similarity function to use
     * @param invert whether to invert the distance (similarity functions have different min/max objectives)
     */
    public VPTree(INDArray items, String similarityFunction, boolean invert) {
        this.similarityFunction = similarityFunction;
        this.invert = invert;
        this.items = items;
        root = buildFromPoints(0, this.items.rows());
    }

    /**
     *
     * @param items the items to use
     * @param similarityFunction the similarity function to use
     * @param parallel whether the tree is parallel o not
     * @param invert whether to invert the metric (different optimization objective)
     */
    public VPTree(List<DataPoint> items,
                  String similarityFunction,
                  boolean parallel,
                  boolean invert) {
        if(this.items == null) {
            this.items = Nd4j.create(items.size());
        }

        this.parallel = parallel;
        for(int i = 0; i < items.size(); i++) {
            this.items.putRow(i,items.get(i).getPoint());
        }

        this.invert = invert;
        this.similarityFunction = similarityFunction;
        root = buildFromPoints(0, items.size());

    }



    /**
     *
     * @param items
     * @param similarityFunction
     */
    public VPTree(INDArray items, String similarityFunction) {
        this(items, similarityFunction, true,true);
    }

    public VPTree(INDArray items,
                  String similarityFunction,
                  boolean parallel,
                  boolean invert) {
        this.similarityFunction = similarityFunction;
        this.invert = invert;
        this.items = items;
        this.parallel = parallel;
        root = buildFromPoints(0, this.items.rows());
    }


    /**
     *
     * @param items
     * @param similarityFunction
     */
    public VPTree(List<DataPoint> items, String similarityFunction) {
        this(items, similarityFunction, true,false);
    }


    /**
     *
     * @param items
     */
    public VPTree(INDArray items) {
        this(items, EUCLIDEAN);
    }


    /**
     *
     * @param items
     */
    public VPTree(List<DataPoint> items) {
        this(items, EUCLIDEAN);
    }

    /**
     * Create an ndarray
     * from the datapoints
     * @param data
     * @return
     */
    public static INDArray buildFromData(List<DataPoint> data) {
        INDArray ret = Nd4j.create(data.size(), data.get(0).getD());
        for (int i = 0; i < ret.slices(); i++)
            ret.putSlice(i, data.get(i).getPoint());
        return ret;
    }



    /**
     *
     * @param basePoint
     * @param distancesArr
     */
    public void calcDistancesRelativeTo(INDArray basePoint,
                                        INDArray distancesArr) {
        switch (similarityFunction) {
            case "euclidean":
                Nd4j.getExecutioner().exec(new EuclideanDistance(items,
                        basePoint,distancesArr,items.length()),1);
                break;
            case "coinedistance":
                Nd4j.getExecutioner().exec(new CosineDistance(items,
                        basePoint,distancesArr,items.length()),1);
                break;
            case "cosinesimilarity":
                Nd4j.getExecutioner().exec(new CosineSimilarity(items,
                        basePoint,distancesArr,items.length()),1);
                break;
            case "manhattan":
                Nd4j.getExecutioner().exec(new ManhattanDistance(items,
                        basePoint,distancesArr,items.length()),1);
                break;
            case "dot":
                Nd4j.getExecutioner().exec(new Dot(items,
                        basePoint,distancesArr,items.length()),1);
                break;
            default:
                Nd4j.getExecutioner().exec(new EuclideanDistance(items,
                        basePoint,distancesArr,items.length()),1);
                break;

        }

        if(invert)
            distancesArr.negi();

    }


    /**
     * Euclidean distance
     * @return the distance between the two points
     */
    public float distance(INDArray arr1,INDArray arr2) {
        switch (similarityFunction) {
            case "euclidean":
                float ret = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr1,arr2))
                        .getFinalResult().floatValue();
                return invert ? -ret : ret;

            case "cosinesimilarity":
                float ret2 = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(arr1,arr2))
                        .getFinalResult().floatValue();
                return invert ? -ret2 : ret2;
            case "cosinedistance":
                float ret6 = Nd4j.getExecutioner().execAndReturn(new CosineDistance(arr1,arr2))
                        .getFinalResult().floatValue();
                return invert ? -ret6 : ret6;

            case "manhattan":
                float ret3 = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(arr1,arr2))
                        .getFinalResult().floatValue();
                return invert ? -ret3 : ret3;
            case "dot":
                float dotRet = (float) Nd4j.getBlasWrapper().dot(arr1,arr2);
                return invert ? -dotRet : dotRet;
            default:
                float ret4 = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr1,arr2))
                        .getFinalResult().floatValue();
                return invert ? -ret4 : ret4;

        }
    }

    private Node buildFromPoints(final int lower, final int upper) {
        if (upper == lower)
            return null;
        if(executorService == null
                &&
                lower == 0 &&
                upper == items.size(0) && parallel)
            executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors(), new ThreadFactory() {
                @Override
                public Thread newThread(Runnable r) {
                    Thread t = Executors.defaultThreadFactory().newThread(r);

                    t.setName("VPTree thread");

                    // we don't want threads to be working on different devices
                    Nd4j.getAffinityManager().attachThreadToDevice(t, Nd4j.getAffinityManager().getDeviceForCurrentThread());

                    return t;
                }
            });

        final Node ret = new Node(lower, 0);
        size.incrementAndGet();

        if (upper - lower > 1) {
            int randomPoint = MathUtils.randomNumberBetween(lower, upper - 1,Nd4j.getRandom());

            // Partition around the median distance
            final int median = (upper + lower) / 2;
            INDArray distancesArr = null;
            INDArray sortedDistances = null;

            if (distancesArr == null)
                distancesArr = Nd4j.create(items.rows(), 1);

            if (sortedDistances == null)
                sortedDistances = Nd4j.create(items.rows(), 1);


            INDArray basePoint = items.getRow(randomPoint);
            //run a distance compute wrt each row given the base point
            calcDistancesRelativeTo(basePoint, distancesArr);

            sortedDistances.assign(distancesArr);

            Nd4j.sort(sortedDistances, 0, false);


            final double medianDistance = sortedDistances.getDouble(sortedDistances.length() / 2);
            INDArray leftPoints = null, rightPoints = null;

            //only allocate left/right points once
            if (leftPoints == null)
                leftPoints = Nd4j.create(sortedDistances.length(), items.columns());

            if (rightPoints == null)
                rightPoints = Nd4j.create(sortedDistances.length(), items.columns());


            int leftPointsIndex = 0;
            int rightPointsIndex = 0;
            synchronized (items) {
                for (int i = 0; i < distancesArr.length(); i++) {
                if (distancesArr.getDouble(i) < medianDistance) {
                    leftPoints.putRow(leftPointsIndex++, items.getRow(i));
                } else {
                    rightPoints.putRow(rightPointsIndex++, items.getRow(i));
                }
            }

                for (int i = 0; i < leftPointsIndex; i++) {
                    items.putRow(i, leftPoints.getRow(i));
                }

                for (int i = 0; i < rightPointsIndex; i++) {
                    items.putRow(i + leftPointsIndex, rightPoints.getRow(i));
                }

                ret.setThreshold(distance(items.getRow(lower), items.getRow(median)));
                ret.setIndex(lower);

            }

            if(parallel && size.get() >= Runtime.getRuntime().availableProcessors()) {
                Future<?> left = null;
                Future<?> right = null;
                if(lower + 1 !=  median) {
                    left = executorService.submit(new Runnable() {
                        @Override
                        public void run() {
                            ret.setLeft(buildFromPoints(lower + 1, median));
                        }
                    });
                }

                if(median != upper) {
                    right = executorService.submit(new Runnable() {
                        @Override
                        public void run() {
                            ret.setRight(buildFromPoints(median, upper));
                        }
                    });
                }

                if(lower == 0 && upper == items.size(0)) {
                    try {
                        if(left != null)
                            left.get();

                        if(right != null)
                            right.get();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    if(executorService != null) {
                        executorService.shutdown();
                    }
                }


            }

            else {
                //no parallel (mainly for debugging)
                if (lower + 1 != median) {
                    ret.setLeft(buildFromPoints(lower + 1, median));

                }

                if (median != upper) {
                    ret.setRight(buildFromPoints(median, upper));

                }

            }

        }


        return ret;

    }



    /**
     *
     * @param target
     * @param k
     * @param results
     * @param distances
     */
    public void search(INDArray target,
                       int k,
                       List<DataPoint> results,
                       List<Double> distances) {
        k = Math.min(k,items.rows());
        results.clear();
        distances.clear();

        PriorityQueue<HeapItem> pq = new PriorityQueue<>();
        tau =  Double.MAX_VALUE;
        search(root, target, k, pq);


        while (!pq.isEmpty()) {
            int idx = pq.peek().getIndex();
            results.add(new DataPoint(idx,items.getRow(idx)));
            distances.add(pq.peek().getDistance());
            pq.next();
        }


        if (invert) {
            Collections.reverse(results);
            Collections.reverse(distances);
        }
    }

    /**
     *
     * @param node
     * @param target
     * @param k
     * @param pq
     */
    public void search(Node node,
                       INDArray target,
                       int k,
                       PriorityQueue<HeapItem> pq) {

        if (node == null)
            return;

        INDArray get = items.getRow(node.getIndex());
        double distance = distance(get, target);
        if (distance < tau) {
            if (pq.size() == k)
                pq.next();
            pq.add(new HeapItem(node.index, distance), distance);
            if (pq.size() == k)
                tau = pq.peek().getDistance();


        }

        Node left = node.getLeft();
        Node right = node.getRight();

        if (left == null && right == null)
            return;

        if (distance < node.getThreshold()) {
            if (distance - tau <= node.getThreshold()) { // if there can still be neighbors inside the ball, recursively search left child first
                search(left, target, k, pq);
            }

             if (distance + tau >= node.getThreshold()) { // if there can still be neighbors outside the ball, recursively search right child
                search(right, target, k, pq);
            }

        } else {
            if (distance + tau >= node.getThreshold()) { // if there can still be neighbors outside the ball, recursively search right child first
                search(right, target, k, pq);
            }

             if (distance - tau <= node.getThreshold()) { // if there can still be neighbors inside the ball, recursively search left child
                search(left, target, k, pq);
            }
        }

    }


    @Data
    public static class Node {
        private int index;
        private float threshold;
        private Node left, right;

        public Node(int index, float threshold) {
            this.index = index;
            this.threshold = threshold;
        }

    }

}
