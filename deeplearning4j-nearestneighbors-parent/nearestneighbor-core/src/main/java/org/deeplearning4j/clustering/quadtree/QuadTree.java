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

package org.deeplearning4j.clustering.quadtree;

import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

import static java.lang.Math.max;

/**
 * QuadTree: http://en.wikipedia.org/wiki/Quadtree
 *
 * Reference impl based on the paper by:
 * http://arxiv.org/pdf/1301.3342v2.pdf
 *
 * Primarily focused on 2 dimensions, may expand later if there's a reason.
 *
 * @author Adam Gibson
 */
public class QuadTree implements Serializable {
    private QuadTree parent, northWest, northEast, southWest, southEast;
    private boolean isLeaf = true;
    private int size, cumSize;
    private Cell boundary;
    static final int QT_NO_DIMS = 2;
    static final int QT_NODE_CAPACITY = 1;
    private INDArray buf = Nd4j.create(QT_NO_DIMS);
    private INDArray data, centerOfMass = Nd4j.create(QT_NO_DIMS);
    private int[] index = new int[QT_NODE_CAPACITY];


    /**
     * Pass in a matrix
     * @param data
     */
    public QuadTree(INDArray data) {
        INDArray meanY = data.mean(0);
        INDArray minY = data.min(0);
        INDArray maxY = data.max(0);
        init(data, meanY.getDouble(0), meanY.getDouble(1),
                        max(maxY.getDouble(0) - meanY.getDouble(0), meanY.getDouble(0) - minY.getDouble(0))
                                        + Nd4j.EPS_THRESHOLD,
                        max(maxY.getDouble(1) - meanY.getDouble(1), meanY.getDouble(1) - minY.getDouble(1))
                                        + Nd4j.EPS_THRESHOLD);
        fill();
    }

    public QuadTree(QuadTree parent, INDArray data, Cell boundary) {
        this.parent = parent;
        this.boundary = boundary;
        this.data = data;

    }

    public QuadTree(Cell boundary) {
        this.boundary = boundary;
    }

    private void init(INDArray data, double x, double y, double hw, double hh) {
        boundary = new Cell(x, y, hw, hh);
        this.data = data;
    }

    private void fill() {
        for (int i = 0; i < data.rows(); i++)
            insert(i);
    }



    /**
     * Returns the cell of this element
     *
     * @param coordinates
     * @return
     */
    protected QuadTree findIndex(INDArray coordinates) {

        // Compute the sector for the coordinates
        boolean left = (coordinates.getDouble(0) <= (boundary.getX() + boundary.getHw() / 2));
        boolean top = (coordinates.getDouble(1) <= (boundary.getY() + boundary.getHh() / 2));

        // top left
        QuadTree index = getNorthWest();
        if (left) {
            // left side
            if (!top) {
                // bottom left
                index = getSouthWest();
            }
        } else {
            // right side
            if (top) {
                // top right
                index = getNorthEast();
            } else {
                // bottom right
                index = getSouthEast();

            }
        }

        return index;
    }


    /**
     * Insert an index of the data in to the tree
     * @param newIndex the index to insert in to the tree
     * @return whether the index was inserted or not
     */
    public boolean insert(int newIndex) {
        // Ignore objects which do not belong in this quad tree
        INDArray point = data.slice(newIndex);
        if (!boundary.containsPoint(point))
            return false;

        cumSize++;
        double mult1 = (double) (cumSize - 1) / (double) cumSize;
        double mult2 = 1.0 / (double) cumSize;

        centerOfMass.muli(mult1);
        centerOfMass.addi(point.mul(mult2));

        // If there is space in this quad tree and it is a leaf, add the object here
        if (isLeaf() && size < QT_NODE_CAPACITY) {
            index[size] = newIndex;
            size++;
            return true;
        }

        //duplicate point
        if (size > 0) {
            for (int i = 0; i < size; i++) {
                INDArray compPoint = data.slice(index[i]);
                if (point.getDouble(0) == compPoint.getDouble(0) && point.getDouble(1) == compPoint.getDouble(1))
                    return true;
            }
        }



        // If this Node has already been subdivided just add the elements to the
        // appropriate cell
        if (!isLeaf()) {
            QuadTree index = findIndex(point);
            index.insert(newIndex);
            return true;
        }

        if (isLeaf())
            subDivide();

        boolean ret = insertIntoOneOf(newIndex);



        return ret;
    }

    private boolean insertIntoOneOf(int index) {
        boolean success = false;
        success = northWest.insert(index);
        if (!success)
            success = northEast.insert(index);
        if (!success)
            success = southWest.insert(index);
        if (!success)
            success = southEast.insert(index);
        return success;
    }


    /**
     * Returns whether the tree is consistent or not
     * @return whether the tree is consistent or not
     */
    public boolean isCorrect() {

        for (int n = 0; n < size; n++) {
            INDArray point = data.slice(index[n]);
            if (!boundary.containsPoint(point))
                return false;
        }

        return isLeaf() || northWest.isCorrect() && northEast.isCorrect() && southWest.isCorrect()
                        && southEast.isCorrect();

    }



    /**
     *  Create four children
     *  which fully divide this cell
     *  into four quads of equal area
     */
    public void subDivide() {
        northWest = new QuadTree(this, data, new Cell(boundary.getX() - .5 * boundary.getHw(),
                        boundary.getY() - .5 * boundary.getHh(), .5 * boundary.getHw(), .5 * boundary.getHh()));
        northEast = new QuadTree(this, data, new Cell(boundary.getX() + .5 * boundary.getHw(),
                        boundary.getY() - .5 * boundary.getHh(), .5 * boundary.getHw(), .5 * boundary.getHh()));
        southWest = new QuadTree(this, data, new Cell(boundary.getX() - .5 * boundary.getHw(),
                        boundary.getY() + .5 * boundary.getHh(), .5 * boundary.getHw(), .5 * boundary.getHh()));
        southEast = new QuadTree(this, data, new Cell(boundary.getX() + .5 * boundary.getHw(),
                        boundary.getY() + .5 * boundary.getHh(), .5 * boundary.getHw(), .5 * boundary.getHh()));

    }


    /**
     * Compute non edge forces using barnes hut
     * @param pointIndex
     * @param theta
     * @param negativeForce
     * @param sumQ
     */
    public void computeNonEdgeForces(int pointIndex, double theta, INDArray negativeForce, AtomicDouble sumQ) {
        // Make sure that we spend no time on empty nodes or self-interactions
        if (cumSize == 0 || (isLeaf() && size == 1 && index[0] == pointIndex))
            return;


        // Compute distance between point and center-of-mass
        buf.assign(data.slice(pointIndex)).subi(centerOfMass);

        double D = Nd4j.getBlasWrapper().dot(buf, buf);

        // Check whether we can use this node as a "summary"
        if (isLeaf || FastMath.max(boundary.getHh(), boundary.getHw()) / FastMath.sqrt(D) < theta) {

            // Compute and add t-SNE force between point and current node
            double Q = 1.0 / (1.0 + D);
            double mult = cumSize * Q;
            sumQ.addAndGet(mult);
            mult *= Q;
            negativeForce.addi(buf.mul(mult));

        } else {

            // Recursively apply Barnes-Hut to children
            northWest.computeNonEdgeForces(pointIndex, theta, negativeForce, sumQ);
            northEast.computeNonEdgeForces(pointIndex, theta, negativeForce, sumQ);
            southWest.computeNonEdgeForces(pointIndex, theta, negativeForce, sumQ);
            southEast.computeNonEdgeForces(pointIndex, theta, negativeForce, sumQ);
        }
    }



    /**
     *
     * @param rowP a vector
     * @param colP
     * @param valP
     * @param N
     * @param posF
     */
    public void computeEdgeForces(INDArray rowP, INDArray colP, INDArray valP, int N, INDArray posF) {
        if (!rowP.isVector())
            throw new IllegalArgumentException("RowP must be a vector");

        // Loop over all edges in the graph
        double D;
        for (int n = 0; n < N; n++) {
            for (int i = rowP.getInt(n); i < rowP.getInt(n + 1); i++) {

                // Compute pairwise distance and Q-value
                buf.assign(data.slice(n)).subi(data.slice(colP.getInt(i)));

                D = Nd4j.getBlasWrapper().dot(buf, buf);
                D = valP.getDouble(i) / D;

                // Sum positive force
                posF.slice(n).addi(buf.mul(D));

            }
        }
    }


    /**
     * The depth of the node
     * @return the depth of the node
     */
    public int depth() {
        if (isLeaf())
            return 1;
        return 1 + max(max(northWest.depth(), northEast.depth()), max(southWest.depth(), southEast.depth()));
    }

    public INDArray getCenterOfMass() {
        return centerOfMass;
    }

    public void setCenterOfMass(INDArray centerOfMass) {
        this.centerOfMass = centerOfMass;
    }

    public QuadTree getParent() {
        return parent;
    }

    public void setParent(QuadTree parent) {
        this.parent = parent;
    }

    public QuadTree getNorthWest() {
        return northWest;
    }

    public void setNorthWest(QuadTree northWest) {
        this.northWest = northWest;
    }

    public QuadTree getNorthEast() {
        return northEast;
    }

    public void setNorthEast(QuadTree northEast) {
        this.northEast = northEast;
    }

    public QuadTree getSouthWest() {
        return southWest;
    }

    public void setSouthWest(QuadTree southWest) {
        this.southWest = southWest;
    }

    public QuadTree getSouthEast() {
        return southEast;
    }

    public void setSouthEast(QuadTree southEast) {
        this.southEast = southEast;
    }

    public boolean isLeaf() {
        return isLeaf;
    }

    public void setLeaf(boolean isLeaf) {
        this.isLeaf = isLeaf;
    }

    public int getSize() {
        return size;
    }

    public void setSize(int size) {
        this.size = size;
    }

    public int getCumSize() {
        return cumSize;
    }

    public void setCumSize(int cumSize) {
        this.cumSize = cumSize;
    }

    public Cell getBoundary() {
        return boundary;
    }

    public void setBoundary(Cell boundary) {
        this.boundary = boundary;
    }
}
