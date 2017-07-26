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
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.clustering.berkeley.PriorityQueue;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.sptree.HeapItem;
import org.deeplearning4j.clustering.sptree.HeapObject;
import org.deeplearning4j.clustering.util.MathUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.Dot;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.CosineSimilarity;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.api.ops.impl.accum.distances.ManhattanDistance;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * Vantage point tree implementation
 *
 * @author Adam Gibson
 */
@Slf4j
@Builder
@AllArgsConstructor
public class VPTree {

    public static final String EUCLIDEAN = "euclidean";
    private double tau;
    @Getter
    @Setter
    private INDArray items;
    private List<INDArray> itemsList;
    private Node root;
    private String similarityFunction;
    @Getter
    private boolean invert = false;
    private ExecutorService executorService;
    @Getter
    private int workers = 2;
    private AtomicInteger size = new AtomicInteger(0);

    private ThreadLocal<INDArray> scalars = new ThreadLocal<>();

    WorkspaceConfiguration workspaceConfiguration;

    /**
     *
     * @param points
     * @param invert
     */
    public VPTree(INDArray points, boolean invert) {
        this(points, "euclidean", 2, invert);
    }

    /**
     *
     * @param points
     * @param invert
     * @param workers number of parallel workers for tree building (increases memory requirements!)
     */
    public VPTree(INDArray points, boolean invert, int workers) {
        this(points, "euclidean", workers, invert);
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
//        itemsList = new ArrayList<>(items.rows());
//        for(int i = 0; i < items.rows(); i++) {
//            itemsList.add(items.getRow(i));
//        }
        root = buildFromPoints(items);
    }

    /**
     *
     * @param items the items to use
     * @param similarityFunction the similarity function to use
     * @param workers number of parallel workers for tree building (increases memory requirements!)
     * @param invert whether to invert the metric (different optimization objective)
     */
    public VPTree(List<DataPoint> items, String similarityFunction, int workers, boolean invert) {
        if (this.items == null) {
            this.items = Nd4j.create(items.size(),items.get(0).getPoint().columns());
        }

        this.workers = workers;

//        this.parallel = parallel;
        for (int i = 0; i < items.size(); i++) {
            //itemsList.add(items.get(i).getPoint());
            this.items.putRow(i, items.get(i).getPoint());
        }

//        itemsList = new ArrayList<>(items.size());
//        for (int i = 0; i < items.size(); i++) {
//            itemsList.add(items.get(i).getPoint());
//        }

        this.invert = invert;
        this.similarityFunction = similarityFunction;
        root = buildFromPoints(this.items);

    }



    /**
     *
     * @param items
     * @param similarityFunction
     */
    public VPTree(INDArray items, String similarityFunction) {
        this(items, similarityFunction, 2, true);
    }

    /**
     *
     * @param items
     * @param similarityFunction
     * @param workers number of parallel workers for tree building (increases memory requirements!)
     * @param invert
     */
    public VPTree(INDArray items, String similarityFunction, int workers, boolean invert) {
        this.similarityFunction = similarityFunction;
        this.invert = invert;
        this.items = items;
//        itemsList = new ArrayList<>(items.rows());
//        for (int i = 0; i < items.rows(); i++) {
//            itemsList.add(items.getRow(i));
//        }

        this.workers = workers;
        root = buildFromPoints(items);
    }


    /**
     *
     * @param items
     * @param similarityFunction
     */
    public VPTree(List<DataPoint> items, String similarityFunction) {
        this(items, similarityFunction, 2, false);
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
    public void calcDistancesRelativeTo(INDArray items, INDArray basePoint, INDArray distancesArr) {
        switch (similarityFunction) {
            case "euclidean":
                Nd4j.getExecutioner().exec(new EuclideanDistance(items, basePoint, distancesArr, items.lengthLong()), -1);
                break;
            case "cosinedistance":
                Nd4j.getExecutioner().exec(new CosineDistance(items, basePoint, distancesArr, items.lengthLong()), -1);
                break;
            case "cosinesimilarity":
                Nd4j.getExecutioner().exec(new CosineSimilarity(items, basePoint, distancesArr, items.lengthLong()), -1);
                break;
            case "manhattan":
                Nd4j.getExecutioner().exec(new ManhattanDistance(items, basePoint, distancesArr, items.lengthLong()), -1);
                break;
            case "dot":
                Nd4j.getExecutioner().exec(new Dot(items, basePoint, distancesArr, items.lengthLong()), -1);
                break;
            default:
                Nd4j.getExecutioner().exec(new EuclideanDistance(items, basePoint, distancesArr, items.lengthLong()), -1);
                break;

        }

        if (invert)
            distancesArr.negi();

    }

    public void calcDistancesRelativeTo(INDArray basePoint, INDArray distancesArr) {
        calcDistancesRelativeTo(items, basePoint, distancesArr);
    }


    /**
     * Euclidean distance
     * @return the distance between the two points
     */
    public float distance(INDArray arr1, INDArray arr2) {
        if (scalars.get() == null)
            scalars.set(Nd4j.scalar(0.0));

        switch (similarityFunction) {
            case "euclidean":
                float ret = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr1, arr2, scalars.get(),arr1.length())).getFinalResult().floatValue();
                return invert ? -ret : ret;
            case "cosinesimilarity":
                float ret2 = Nd4j.getExecutioner().execAndReturn(new CosineSimilarity(arr1, arr2, scalars.get(), arr1.length())).getFinalResult().floatValue();
                return invert ? -ret2 : ret2;
            case "cosinedistance":
                float ret6 = Nd4j.getExecutioner().execAndReturn(new CosineDistance(arr1, arr2, scalars.get(), arr1.length())).getFinalResult().floatValue();
                return invert ? -ret6 : ret6;
            case "manhattan":
                float ret3 = Nd4j.getExecutioner().execAndReturn(new ManhattanDistance(arr1, arr2,scalars.get(),arr1.length())).getFinalResult().floatValue();
                return invert ? -ret3 : ret3;
            case "dot":
                float dotRet = (float) Nd4j.getBlasWrapper().dot(arr1, arr2);
                return invert ? -dotRet : dotRet;
            default:
                float ret4 = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(arr1, arr2, scalars.get(),arr1.length())).getFinalResult()
                        .floatValue();
                return invert ? -ret4 : ret4;

        }
    }

    protected class NodeBuilder implements Callable<Node> {
        protected List<INDArray> list;
        protected List<Integer> indices;
        public NodeBuilder(List<INDArray> list, List<Integer> indices) {
            this.list = list;
            this.indices = indices;
        }

        @Override
        public Node call() throws Exception {
            return buildFromPoints(list, indices);
        }
    }

    private Node buildFromPoints(List<INDArray> points, List<Integer> indices) {
        Node ret = new Node(0, 0);


        // nothing to sort here
        if (points.size() == 1) {
            ret.point = points.get(0);
            ret.index = indices.get(0);
            return ret;
        }

        // opening workspace, and creating it if that's the first call
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfiguration, "VPTREE_WORSKPACE");

        INDArray items = Nd4j.vstack(points);
        int randomPoint = MathUtils.randomNumberBetween(0, items.rows() - 1, Nd4j.getRandom());
        INDArray basePoint = points.get(randomPoint);//items.getRow(randomPoint);
        ret.point = basePoint;
        ret.index = indices.get(randomPoint);
        INDArray distancesArr = Nd4j.create(items.rows(), 1);

        calcDistancesRelativeTo(items, basePoint, distancesArr);

        double medianDistance = distancesArr.medianNumber().doubleValue();

        ret.threshold = (float) medianDistance;

        List<INDArray> leftPoints = new ArrayList<>();
        List<Integer> leftIndices = new ArrayList<>();
        List<INDArray> rightPoints = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();

        for (int i = 0; i < distancesArr.length(); i++) {
            if (i == randomPoint)
                continue;

            if (distancesArr.getDouble(i) < medianDistance) {
                leftPoints.add(points.get(i));
                leftIndices.add(indices.get(i));
            } else {
                rightPoints.add(points.get(i));
                rightIndices.add(indices.get(i));
            }
        }

        // closing workspace
        workspace.notifyScopeLeft();

        if (leftPoints.size() > 0)
            ret.futureLeft = executorService.submit(new NodeBuilder(leftPoints, leftIndices)); // = buildFromPoints(leftPoints);

        if (rightPoints.size() > 0)
            ret.futureRight = executorService.submit(new NodeBuilder(rightPoints, rightIndices));

        //System.gc();
        return ret;
    }

    private Node buildFromPoints(INDArray items) {
        if (executorService == null && items == this.items)
            executorService = Executors.newFixedThreadPool( workers,
                    new ThreadFactory() {
                        @Override
                        public Thread newThread(Runnable r) {
                            Thread t = Executors.defaultThreadFactory().newThread(r);

                            t.setDaemon(true);
                            t.setName("VPTree thread");

                            // we don't want threads to be working on different devices
                            Nd4j.getAffinityManager().attachThreadToDevice(t,
                                    Nd4j.getAffinityManager().getDeviceForCurrentThread());

                            return t;
                        }
                    });

        final Node ret = new Node(0, 0);
        size.incrementAndGet();

        workspaceConfiguration = WorkspaceConfiguration.builder()
                .cyclesBeforeInitialization(1)
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.FULL)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .build();

        // opening workspace
        MemoryWorkspace workspace = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfiguration, "VPTREE_WORSKPACE");

        int randomPoint = MathUtils.randomNumberBetween(0, items.rows() - 1, Nd4j.getRandom());
        INDArray basePoint = items.getRow(randomPoint);
        INDArray distancesArr = Nd4j.create(items.rows(), 1);
        ret.point = basePoint;
        ret.index = randomPoint;

        calcDistancesRelativeTo(items, basePoint, distancesArr);

        double medianDistance = distancesArr.medianNumber().doubleValue();

        ret.threshold = (float) medianDistance;

        List<INDArray> leftPoints = new ArrayList<>();
        List<Integer> leftIndices = new ArrayList<>();
        List<INDArray> rightPoints = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();

        for (int i = 0; i < distancesArr.length(); i++) {
            if (i == randomPoint)
                continue;

            if (distancesArr.getDouble(i) < medianDistance) {
                leftPoints.add(items.getRow(i));
                leftIndices.add(i);
            } else {
                rightPoints.add(items.getRow(i));
                rightIndices.add(i);
            }
        }

        // closing workspace
        workspace.notifyScopeLeft();
        workspace.destroyWorkspace(true);

        if (leftPoints.size() > 0)
            ret.left = buildFromPoints(leftPoints, leftIndices);

        if (rightPoints.size() > 0)
            ret.right = buildFromPoints(rightPoints, rightIndices);

        // destroy once again
        workspace.destroyWorkspace(true);

        if (ret.left != null)
            ret.left.fetchFutures();

        if (ret.right != null)
            ret.right.fetchFutures();

        if (executorService != null)
            executorService.shutdown();

        return ret;
    }



    /**
     *
     * @param target
     * @param k
     * @param results
     * @param distances
     */
    public void search(INDArray target, int k, List<DataPoint> results, List<Double> distances) {
        k = Math.min(k, items.rows());
        results.clear();
        distances.clear();

        PriorityQueue<HeapObject> pq = new PriorityQueue<>();
        search(root, target, k+1, pq,  Double.MAX_VALUE);

        if (pq.size() > k)
            pq.next();

        while (!pq.isEmpty()) {
            HeapObject ho = pq.peek();
            results.add(new DataPoint(ho.getIndex(), ho.getPoint()));
            distances.add(ho.getDistance());
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
    public void search(Node node, INDArray target, int k, PriorityQueue<HeapObject> pq, double cTau) {

        if (node == null)
            return;

        double tau = cTau;

        INDArray get = node.getPoint(); //items.getRow(node.getIndex());
        double distance = distance(get, target);
        if (distance < tau) {
            if (pq.size() == k)
                pq.next();
            pq.add(new HeapObject(node.getIndex(), node.getPoint(), distance), distance);
            if (pq.size() == k)
                tau = pq.peek().getDistance();
        }

        Node left = node.getLeft();
        Node right = node.getRight();

        if (left == null && right == null)
            return;

        if (distance < node.getThreshold()) {
            if (distance - tau < node.getThreshold()) { // if there can still be neighbors inside the ball, recursively search left child first
                search(left, target, k, pq, tau);
            }

            if (distance + tau >= node.getThreshold()) { // if there can still be neighbors outside the ball, recursively search right child
                search(right, target, k, pq, tau);
            }

        } else {
            if (distance + tau >= node.getThreshold()) { // if there can still be neighbors outside the ball, recursively search right child first
                search(right, target, k, pq, tau);
            }

            if (distance - tau < node.getThreshold()) { // if there can still be neighbors inside the ball, recursively search left child
                search(left, target, k, pq, tau);
            }
        }

    }


    @Data
    public static class Node {
        private int index;
        private float threshold;
        private Node left, right;
        private INDArray point;
        protected Future<Node> futureLeft;
        protected Future<Node> futureRight;

        public Node(int index, float threshold) {
            this.index = index;
            this.threshold = threshold;
        }


        public void fetchFutures() {
            try {
                if (futureLeft != null) {
                    while (!futureLeft.isDone())
                        Thread.sleep(100);


                    left = futureLeft.get();
                }

                if (futureRight != null) {
                    while (!futureRight.isDone())
                        Thread.sleep(100);

                    right = futureRight.get();
                }


                if (left != null)
                    left.fetchFutures();

                if (right != null)
                    right.fetchFutures();
            } catch (Exception e) {
                throw new RuntimeException(e);
            }


        }
    }

}
