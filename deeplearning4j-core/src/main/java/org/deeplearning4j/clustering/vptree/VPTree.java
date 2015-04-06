package org.deeplearning4j.clustering.vptree;

import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

/**
 * @author Adam Gibson
 */
public class VPTree {

    private List<DataPoint> items;
    private double tau;

    private Node buildFromPoints(int lower,int upper) {
        if(upper == lower)
            return null;
        Node ret = new Node(lower,0);
        if(upper - lower > 1) {
            int i = MathUtils.randomNumberBetween(upper, lower);

            // Nd4j.getBlasWrapper().swap(items.get(lower),items.slice(i));
            // Partition around the median distance
            int median = (upper + lower) / 2;


        }

        return ret;

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
