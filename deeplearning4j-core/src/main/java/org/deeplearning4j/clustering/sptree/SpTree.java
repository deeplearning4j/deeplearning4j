package org.deeplearning4j.clustering.sptree;

import com.google.common.util.concurrent.AtomicDouble;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;


/**
 * @author Adam Gibson
 */
public class SpTree implements Serializable {
    private int D;
    public final static int QT_NODE_CAPACITY = 4;

    private INDArray data;
    private int N;
    private INDArray buf;
    private int size;
    private int cumSize;
    private Cell boundary;
    private INDArray centerOfMass;
    private SpTree parent;
    private int[] index = new int[QT_NODE_CAPACITY];
    private int numChildren = 2;
    private boolean isLeaf = true;
    private Set<INDArray> indices;
    private SpTree[] children;




    public SpTree(SpTree parent,int D,INDArray data,INDArray corner,INDArray width,Set<INDArray> indices) {
        init(parent, D, data, corner, width,indices);
    }


    public SpTree(int d, INDArray data, int n,Set<INDArray> indices) {
        this.D = d;
        this.data = data;
        N = n;
        this.indices = indices;
        INDArray meanY = data.mean(0);
        INDArray minY = data.min(0);
        INDArray maxY = data.max(0);
        INDArray width = Nd4j.create(meanY.shape());
        for(int i = 0; i < width.length(); i++) {
            width.putScalar(i, FastMath.max(maxY.getDouble(i) - meanY.getDouble(i),meanY.getDouble(i) - minY.getDouble(i) + Nd4j.EPS_THRESHOLD));
        }

        init(null,D,data,meanY,width,indices);
        fill(N);


    }



    public SpTree(int d, INDArray data, int n) {
        this(d,data,n,new HashSet<INDArray>());
    }

    private void init(SpTree parent,int D,INDArray data,INDArray corner,INDArray width,Set<INDArray> indices) {
        this.parent = parent;
        this.D = D;
        for(int d = 1; d < this.D; d++)
            numChildren *= 2;
        this.indices = indices;
        isLeaf = true;
        size = 0;
        cumSize = 0;
        children = new SpTree[numChildren];
        this.data = data;
        boundary = new Cell(D);
        boundary.setCorner(corner.dup());
        boundary.setWidth(width.dup());
        centerOfMass = Nd4j.create(D);
        buf = Nd4j.create(D);
    }




    private boolean insert(int index) {
        INDArray point = data.slice(index);
        if(!boundary.contains(point))
            return false;


        cumSize++;
        double mult1 = (double) (cumSize - 1) / (double) cumSize;
        double mult2 = 1.0 / (double) cumSize;
        centerOfMass.muli(mult1);
        centerOfMass.addi(point.mul(mult2));
        // If there is space in this quad tree and it is a leaf, add the object here
        if(isLeaf() && size < QT_NODE_CAPACITY) {
            this.index[size] = index;
            size++;
            return true;
        }


        for(int i = 0; i < size; i++) {
            INDArray compPoint = data.slice(this.index[i]);
            if(compPoint.equals(point))
                return true;
        }



        if(isLeaf())
            subDivide();


        // Find out where the point can be inserted
        for(int i = 0; i < numChildren; i++) {
            if(children[i].insert(index))
                return true;
        }

        return false;
    }


    /**
     * Subdivide the node in to
     * 4 children
     */
    public void subDivide() {
        INDArray newCorner = Nd4j.create(D);
        INDArray newWidth = Nd4j.create(D);
        for( int i = 0; i < numChildren; i++) {
            int div = 1;
            for( int d = 0; d < D; d++) {
                newWidth.putScalar(d,.5 * boundary.width(d));
                if((i / div) % 2 == 1)
                    newCorner.putScalar(d, boundary.corner(d) - .5 * boundary.width(d));
                else
                    newCorner.putScalar(d,boundary.corner(d) + .5 * boundary.width(d));
                div *= 2;
            }

            children[i] = new SpTree(this, D, data, newCorner, newWidth,indices);

        }

        // Move existing points to correct children
        for(int i = 0; i < size; i++) {
            boolean success = false;
            for(int j = 0; j < this.numChildren; j++)
                if(!success)
                    success = children[j].insert(index[i]);

            index[i] = -1;
        }

        // Empty parent node
        size = 0;
        isLeaf = false;
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
        if(cumSize == 0 || (isLeaf() && size == 1 && index[0] == pointIndex))
            return;


        // Compute distance between point and center-of-mass
        buf.assign(data.slice(pointIndex)).subi(centerOfMass);

        double D = Nd4j.getBlasWrapper().dot(buf, buf);
        // Check whether we can use this node as a "summary"
        double maxWidth = boundary.width().max(Integer.MAX_VALUE).getDouble(0);
        // Check whether we can use this node as a "summary"
        if(isLeaf() || maxWidth / FastMath.sqrt(D) < theta) {

            // Compute and add t-SNE force between point and current node
            double Q = 1.0 / (1.0 + D);
            double mult = cumSize * Q;
            sumQ.addAndGet(mult);
            mult *= Q;
            negativeForce.addi(buf.mul(mult));

        }
        else {

            // Recursively apply Barnes-Hut to children
            for(int i = 0; i < numChildren; i++) {
                children[i].computeNonEdgeForces(pointIndex, theta, negativeForce, sumQ);
            }

        }
    }


    /**
     *
     * Compute edge forces using barns hut
     * @param rowP a vector
     * @param colP
     * @param valP
     * @param N the number of elements
     * @param posF the positive force
     */
    public void computeEdgeForces(INDArray rowP, INDArray colP, INDArray valP, int N, INDArray posF) {
        if(!rowP.isVector())
            throw new IllegalArgumentException("RowP must be a vector");

        // Loop over all edges in the graph
        double D;
        for(int n = 0; n < N; n++) {
            for(int i = rowP.getInt(n); i < rowP.getInt(n + 1); i++) {

                // Compute pairwise distance and Q-value
                buf.assign(data.slice(n)).subi(data.slice(colP.getInt(i)));

                D = Nd4j.getBlasWrapper().dot(buf,buf);
                D = valP.getDouble(i) / D;

                // Sum positive force
                posF.slice(n).addi(buf.mul(D));

            }
        }
    }



    public boolean isLeaf() {
        return isLeaf;
    }

    /**
     * Verifies the structure of the tree (does bounds checking on each node)
     * @return true if the structure of the tree
     * is correct.
     */
    public boolean isCorrect() {
        for(int n = 0; n < size; n++) {
            INDArray point = data.slice(index[n]);
            if(!boundary.contains(point))
                return false;
        }
        if(!isLeaf()) {
            boolean correct = true;
            for(int i = 0; i < numChildren; i++)
                correct = correct && children[i].isCorrect();
            return correct;
        }

        return true;
    }

    /**
     * The depth of the node
     * @return the depth of the node
     */
    public int depth() {
        if(isLeaf())
            return 1;
        int depth = 1;
        int maxChildDepth = 0;
        for(int i = 0; i < numChildren; i++) {
            maxChildDepth = Math.max(maxChildDepth, children[0].depth());
        }

        return depth + maxChildDepth;
    }

    private void fill(int n) {
        for(int i = 0; i < n; i++)
            insert(i);
    }


    public SpTree[] getChildren() {
        return children;
    }

    public int getD() {
        return D;
    }

    public INDArray getCenterOfMass() {
        return centerOfMass;
    }

    public Cell getBoundary() {
        return boundary;
    }

    public int[] getIndex() {
        return index;
    }

    public int getCumSize() {
        return cumSize;
    }

    public void setCumSize(int cumSize) {
        this.cumSize = cumSize;
    }

    public int getNumChildren() {
        return numChildren;
    }

    public void setNumChildren(int numChildren) {
        this.numChildren = numChildren;
    }

}
