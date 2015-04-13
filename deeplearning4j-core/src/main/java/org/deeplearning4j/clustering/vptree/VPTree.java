package org.deeplearning4j.clustering.vptree;

import org.deeplearning4j.berkeley.PriorityQueue;
import org.deeplearning4j.clustering.sptree.DataPoint;
import org.deeplearning4j.clustering.sptree.HeapItem;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Vantage point tree implementation
 *
 * @author Adam Gibson
 */
public class VPTree {

    private List<DataPoint> items;
    private double tau;
    private Node root;

    public VPTree(INDArray items) {
        List<DataPoint> thisItems = new ArrayList<>();
        for(int i = 0; i < items.slices(); i++)
            thisItems.add(new DataPoint(i,items.slice(i)));
        this.items = thisItems;
        root = buildFromPoints(0,this.items.size());
    }

    public VPTree(List<DataPoint> items) {
        this.items = items;
        root = buildFromPoints(0,items.size());
    }

    public static INDArray buildFromData(List<DataPoint> data) {
        INDArray ret = Nd4j.create(data.size(),data.get(0).getD());
        for(int i = 0; i < ret.slices(); i++)
            ret.putSlice(i,data.get(i).getPoint());
        return ret;
    }



    private Node buildFromPoints(int lower,int upper) {
        if(upper == lower)
            return null;
        Node ret = new Node(lower,0);
        if(upper - lower > 1) {
            int randomPoint = MathUtils.randomNumberBetween(lower, upper - 1);

            // Nd4j.getBlasWrapper().swap(items.get(lower),items.slice(i));
            // Partition around the median distance
            int median = (upper + lower) / 2;
            double distances[] = new double[items.size()];
            double sortedDistances[] = new double[items.size()];
            DataPoint basePoint = items.get(randomPoint);
            for (int i = 0; i < items.size(); ++i) {
                distances[i] = basePoint.euclidean(items.get(i));
                sortedDistances[i] = distances[i];
            }

            Arrays.sort(sortedDistances);
            final double medianDistance = sortedDistances[sortedDistances.length / 2];
            List<DataPoint> leftPoints = new ArrayList<>(sortedDistances.length);
            List<DataPoint> rightPoints = new ArrayList<>(sortedDistances.length);

            for (int i = 0; i < distances.length; i++) {
                if (distances[i] < medianDistance) {
                    leftPoints.add(items.get(i));
                } else {
                    rightPoints.add(items.get(i));
                }
            }

            for (int i = 0; i < leftPoints.size(); ++i) {
                items.set(i, leftPoints.get(i));
            }

            for (int i = 0; i < rightPoints.size(); ++i) {
                items.set(i + leftPoints.size(), rightPoints.get(i));
            }

            ret.setThreshold(items.get(lower).euclidean(items.get(median)));
            ret.setIndex(lower);
            ret.setRight(buildFromPoints(lower + 1,median));
            ret.setRight(buildFromPoints(median,upper));




        }

        return ret;

    }


    public void search(DataPoint target,int k,List<DataPoint> results,List<Double> distances) {
        PriorityQueue<HeapItem> pq = new PriorityQueue<>();
        tau = Double.MAX_VALUE;
        search(root,target,k,pq);

        results.clear();
        distances.clear();

        while(!pq.isEmpty()) {
            results.add(items.get(pq.peek().getIndex()));
            distances.add(pq.peek().getDistance());
            pq.next();
        }

        Collections.reverse(results);
        Collections.reverse(distances);
    }


    public void search(Node node,DataPoint target,int k,PriorityQueue<HeapItem> pq) {
        if(node == null)
            return;
        double distance = items.get(node.getIndex()).euclidean(target);
        if(distance < tau) {
            if(pq.size() == k)
                pq.next();
                pq.add(new HeapItem(node.index,distance),distance);
                if(pq.size() == k)
                    tau = pq.peek().getDistance();


        }

        if(node.getLeft() == null && node.getRight() == null)
            return;

        if(distance < node.getThreshold()) {
            if(distance - tau <= node.getThreshold()) {         // if there can still be neighbors inside the ball, recursively search left child first
                search(node.getLeft(), target, k, pq);
            }

            if(distance + tau >= node.getThreshold()) {         // if there can still be neighbors outside the ball, recursively search right child
                search(node.getRight(), target, k, pq);
            }

        }
        else {
            if(distance + tau >= node.getThreshold()) {         // if there can still be neighbors outside the ball, recursively search right child first
                search(node.getRight(), target, k, pq);
            }

            if (distance - tau <= node.getThreshold()) {         // if there can still be neighbors inside the ball, recursively search left child
                search(node.getLeft(), target, k, pq);
            }
        }

    }




    public static class Node {
        private int index;
        private double threshold;
        private Node left,right;

        public Node(int index, double threshold) {
            this.index = index;
            this.threshold = threshold;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Node node = (Node) o;

            if (index != node.index) return false;
            if (Double.compare(node.threshold, threshold) != 0) return false;
            if (left != null ? !left.equals(node.left) : node.left != null) return false;
            return !(right != null ? !right.equals(node.right) : node.right != null);

        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            result = index;
            temp = Double.doubleToLongBits(threshold);
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            result = 31 * result + (left != null ? left.hashCode() : 0);
            result = 31 * result + (right != null ? right.hashCode() : 0);
            return result;
        }

        public int getIndex() {
            return index;
        }

        public void setIndex(int index) {
            this.index = index;
        }

        public double getThreshold() {
            return threshold;
        }

        public void setThreshold(double threshold) {
            this.threshold = threshold;
        }

        public Node getLeft() {
            return left;
        }

        public void setLeft(Node left) {
            this.left = left;
        }

        public Node getRight() {
            return right;
        }

        public void setRight(Node right) {
            this.right = right;
        }
    }

}
