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

package org.deeplearning4j.clustering.kdtree;

import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.distances.EuclideanDistance;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * KDTree based on: https://github.com/nicky-zs/kdtree-python/blob/master/kdtree.py
 *
 * @author Adam Gibson
 */
public class KDTree implements Serializable {

    private KDNode root;
    private int dims = 100;
    public final static int GREATER = 1;
    public final static int LESS = 0;
    private int size = 0;
    private HyperRect rect;

    public KDTree(int dims) {
        this.dims = dims;
    }

    /**
     * Insert a point in to the tree
     * @param point the point to insert
     */
    public void insert(INDArray point) {
        if(!point.isVector() || point.length() != dims)
            throw new IllegalArgumentException("Point must be a vector of length " + dims);

        if(root == null) {
            root = new KDNode(point);
            rect = new HyperRect(HyperRect.point(point));
        }
        else {
            int disc = 0;
            KDNode node = root;
            KDNode insert = new KDNode(point);
            int successor;
            while(true) {
                //exactly equal
                if(node.getPoint().eq(point).sum(Integer.MAX_VALUE).getDouble(0) == 0) {
                    return;
                }
                else {
                    successor = successor(root,point,disc);
                    KDNode child;
                    if(successor < 1)
                        child = root.getLeft();
                    else
                        child = root.getRight();
                    if(child == null)
                        break;
                    disc = (disc + 1) % dims;
                    node = child;

                }
            }

            if(successor < 1)
                node.setLeft(insert);

            else
                node.setRight(insert);

            rect.enlargeTo(point);
            insert.setParent(node);
            size++;

        }

    }


    public KDNode delete(INDArray point) {
        KDNode node = root;
        int _disc = 0;
        while(node != null) {
            if(node.point == point)
                break;
            int successor = successor(node,point,_disc);
            if(successor < 1)
                node = node.getLeft();
            else
                node = node.getRight();
            _disc = (_disc + 1) % dims;
        }

        if(node != null) {
            if(node == root) {
                root = delete(root,_disc);
            }
            else
                node = delete(node,_disc);
            size--;
            if(size == 1) {
                rect = new HyperRect(HyperRect.point(point));
            }
            else
                rect = null;

        }
        return node;
    }



    public  List<Pair<Double,INDArray>> knn(INDArray point,double distance) {
        List<Pair<Double,INDArray>> best = new ArrayList<>();
        knn(root,point,rect,distance,best,0);
        Collections.sort(best, new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                return Double.compare(o1.getFirst(),o2.getFirst());
            }
        });

        return best;
    }


    private void knn(KDNode node,INDArray point,HyperRect rect,double dist,List<Pair<Double,INDArray>> best,int _disc) {
        if(node == null || rect.minDistance(point) > dist)
            return;
        int _discNext = (_disc + 1) % dims;
        double distance = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(point)).currentResult().doubleValue();
        if(distance <= dist) {
            best.add(new Pair<>(distance,node.getPoint()));
        }

        HyperRect lower = rect.getLower(point,_disc);
        HyperRect upper = rect.getUpper(point,_disc);
        knn(node.getLeft(),point,lower,dist,best,_discNext);
        knn(node.getRight(),point,upper,dist,best,_discNext);
    }

    /**
     * Query for nearest neighbor. Returns the distance and point
     * @param point the point to query for
     * @return
     */
    public Pair<Double,INDArray> nn(INDArray point) {
        return nn(root,point,rect,Double.POSITIVE_INFINITY,null,0);
    }


    private Pair<Double,INDArray> nn(KDNode node,INDArray point,HyperRect rect,double dist,INDArray best,int _disc) {
        if(node == null || rect.minDistance(point) > dist)
            return new Pair<>(Double.POSITIVE_INFINITY,null);

        int _discNext = (_disc + 1) % dims;
        double dist2 = Nd4j.getExecutioner().execAndReturn(new EuclideanDistance(point)).currentResult().doubleValue();
        if(dist2 < dist) {
            best = node.getPoint();
        }

        HyperRect lower = rect.getLower(node.point,_disc);
        HyperRect upper = rect.getUpper(node.point,_disc);

        if(point.getDouble(_disc) < node.point.getDouble(_disc)) {
            Pair<Double,INDArray> left = nn(node.getLeft(),point,lower,dist,best,_discNext);
            Pair<Double,INDArray> right = nn(node.getRight(),point,upper,dist,best,_discNext);
            if(left.getFirst() < dist)
                return left;
            else if(right.getFirst() < dist)
                return right;

        }
        else {
            Pair<Double,INDArray> left = nn(node.getRight(),point,upper,dist,best,_discNext);
            Pair<Double,INDArray> right = nn(node.getLeft(),point,lower,dist,best,_discNext);
            if(left.getFirst() < dist)
                return left;
            else if(right.getFirst() < dist)
                return right;
        }

        return new Pair<>(dist,best);

    }

    private KDNode delete(KDNode delete,int _disc) {
         if(delete.getLeft() != null && delete.getRight() != null) {
             if(delete.getParent() != null) {
                 if(delete.getParent().getLeft() == delete)
                     delete.getParent().setLeft(null);
                 else
                     delete.getParent().setRight(null);

             }
             return null;
         }

        int disc = _disc;
        _disc = (_disc + 1) % dims;
        Pair<KDNode,Integer> qd = null;
        if(delete.getRight() != null) {
            qd = min(delete.getRight(),disc,_disc);
        }
        else if(delete.getLeft() != null)
            qd = max(delete.getLeft(),disc,_disc);
        delete.point = qd.getFirst().point;
        KDNode qFather = qd.getFirst().getParent();
        if(qFather.getLeft() == qd.getFirst()) {
             qFather.setLeft(delete(qd.getFirst(),disc));
        }
        else if(qFather.getRight() == qd.getFirst()) {
            qFather.setRight(delete(qd.getFirst(), disc));

        }

        return delete;


    }


    private Pair<KDNode,Integer> max(KDNode node,int disc,int _disc) {
        int discNext = (_disc + 1) % dims;
        if(_disc == disc) {
            KDNode child = node.getLeft();
            if(child != null) {
                return max(child,disc,discNext);
            }
        }
        else if(node.getLeft() != null || node.getRight() != null) {
            Pair<KDNode,Integer> left = null,right = null;
            if(node.getLeft() != null)
                left = max(node.getLeft(),disc,discNext);
            if(node.getRight() != null)
                right = max(node.getRight(),disc,discNext);
            if(left != null && right != null) {
                double pointLeft = left.getFirst().getPoint().getDouble(disc);
                double pointRight = right.getFirst().getPoint().getDouble(disc);
                if(pointLeft > pointRight)
                    return left;
                else
                    return right;
            }
            else if(left != null)
                return left;
            else
                return right;
        }

        return new Pair<>(node,_disc);
    }



    private Pair<KDNode,Integer> min(KDNode node,int disc,int _disc) {
        int discNext = (_disc + 1) % dims;
        if(_disc == disc) {
            KDNode child = node.getLeft();
            if(child != null) {
                return min(child,disc,discNext);
            }
        }
        else if(node.getLeft() != null || node.getRight() != null) {
            Pair<KDNode,Integer> left = null,right = null;
            if(node.getLeft() != null)
                left = min(node.getLeft(),disc,discNext);
            if(node.getRight() != null)
                right = min(node.getRight(),disc,discNext);
            if(left != null && right != null) {
                double pointLeft = left.getFirst().getPoint().getDouble(disc);
                double pointRight = right.getFirst().getPoint().getDouble(disc);
                if(pointLeft < pointRight)
                    return left;
                else
                    return right;
            }
            else if(left != null)
                return left;
            else
                return right;
        }

        return new Pair<>(node,_disc);
    }

    /**
     * The number of elements in the tree
     * @return the number of elements in the tree
     */
    public int size() {
        return size;
    }

    private int successor(KDNode node,INDArray point,int disc) {
        for(int i = disc; i < dims; i++) {
            double pointI = point.getDouble(i);
            double nodePointI = node.getPoint().getDouble(i);
            if(pointI < nodePointI)
                return LESS;
            else if(pointI > nodePointI)
                return GREATER;

        }

        throw new IllegalStateException("Point is equal!");
    }


    public static class KDNode {
        private INDArray point;
        private KDNode left,right,parent;

        public KDNode(INDArray point) {
            this.point = point;
        }

        public INDArray getPoint() {
            return point;
        }

        public KDNode getLeft() {
            return left;
        }

        public void setLeft(KDNode left) {
            this.left = left;
        }

        public KDNode getRight() {
            return right;
        }

        public void setRight(KDNode right) {
            this.right = right;
        }

        public KDNode getParent() {
            return parent;
        }

        public void setParent(KDNode parent) {
            this.parent = parent;
        }
    }


}
