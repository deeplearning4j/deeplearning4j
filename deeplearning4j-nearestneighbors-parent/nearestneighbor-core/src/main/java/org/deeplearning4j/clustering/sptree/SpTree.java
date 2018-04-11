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

package org.deeplearning4j.clustering.sptree;

import com.google.common.util.concurrent.AtomicDouble;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;


/**
 * @author Adam Gibson
 */
public class SpTree implements Serializable {


    public final static String workspaceExternal = "SPTREE_LOOP_EXTERNAL";


    private int D;
    private INDArray data;
    public final static int NODE_RATIO = 8000;
    private int N;
    private INDArray buf;
    private int size;
    private int cumSize;
    private Cell boundary;
    private INDArray centerOfMass;
    private SpTree parent;
    private int[] index;
    private int nodeCapacity;
    private int numChildren = 2;
    private boolean isLeaf = true;
    private Set<INDArray> indices;
    private SpTree[] children;
    private static Logger log = LoggerFactory.getLogger(SpTree.class);
    private String similarityFunction = "euclidean";
    protected WorkspaceConfiguration workspaceConfigurationFeedForward = WorkspaceConfiguration.builder().initialSize(0)
            .overallocationLimit(0.2).policyReset(ResetPolicy.BLOCK_LEFT)
            .policyLearning(LearningPolicy.OVER_TIME).policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE).build();

    public final static WorkspaceConfiguration workspaceConfigurationCache = WorkspaceConfiguration.builder()
            .overallocationLimit(0.2).policyReset(ResetPolicy.BLOCK_LEFT).cyclesBeforeInitialization(3)
            .policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.REALLOCATE)
            .policyLearning(LearningPolicy.OVER_TIME).build();

    protected WorkspaceMode workspaceMode;
    protected final static WorkspaceConfiguration workspaceConfigurationExternal = WorkspaceConfiguration.builder()
            .initialSize(0).overallocationLimit(0.3).policyLearning(LearningPolicy.FIRST_LOOP)
            .policyReset(ResetPolicy.BLOCK_LEFT).policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE).build();




    public SpTree(SpTree parent, INDArray data, INDArray corner, INDArray width, Set<INDArray> indices,
                  String similarityFunction) {
        init(parent, data, corner, width, indices, similarityFunction);
    }


    public SpTree(INDArray data, Set<INDArray> indices, String similarityFunction) {
        this.indices = indices;
        this.N = data.rows();
        this.D = data.columns();
        this.similarityFunction = similarityFunction;
        data = data.migrate();
        INDArray meanY = data.mean(0);
        INDArray minY = data.min(0);
        INDArray maxY = data.max(0);
        INDArray width = Nd4j.create(meanY.shape());
        for (int i = 0; i < width.length(); i++) {
            width.putScalar(i, Math.max(maxY.getDouble(i) - meanY.getDouble(i),
                    meanY.getDouble(i) - minY.getDouble(i) + Nd4j.EPS_THRESHOLD));
        }

        init(null, data, meanY, width, indices, similarityFunction);
        fill(N);


    }


    public SpTree(SpTree parent, INDArray data, INDArray corner, INDArray width, Set<INDArray> indices) {
        this(parent, data, corner, width, indices, "euclidean");
    }


    public SpTree(INDArray data, Set<INDArray> indices) {
        this(data, indices, "euclidean");
    }



    public SpTree(INDArray data) {
        this(data, new HashSet<INDArray>());
    }

    public MemoryWorkspace workspace() {
        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);
        return workspace;
    }

    private void init(SpTree parent, INDArray data, INDArray corner, INDArray width, Set<INDArray> indices,
                      String similarityFunction) {

        this.parent = parent;
        D = data.columns();
        N = data.rows();
        this.similarityFunction = similarityFunction;
        nodeCapacity = N % NODE_RATIO;
        index = new int[nodeCapacity];
        for (int d = 1; d < this.D; d++)
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
        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {

            INDArray point = data.slice(index);
            if (!boundary.contains(point))
                return false;


            cumSize++;
            double mult1 = (double) (cumSize - 1) / (double) cumSize;
            double mult2 = 1.0 / (double) cumSize;
            centerOfMass.muli(mult1);
            centerOfMass.addi(point.mul(mult2));
            // If there is space in this quad tree and it is a leaf, add the object here
            if (isLeaf() && size < nodeCapacity) {
                this.index[size] = index;
                indices.add(point);
                size++;
                return true;
            }


            for (int i = 0; i < size; i++) {
                INDArray compPoint = data.slice(this.index[i]);
                if (compPoint.equals(point))
                    return true;
            }


            if (isLeaf())
                subDivide();


            // Find out where the point can be inserted
            for (int i = 0; i < numChildren; i++) {
                if (children[i].insert(index))
                    return true;
            }

            throw new IllegalStateException("Shouldn't reach this state");
        }
    }


    /**
     * Subdivide the node in to
     * 4 children
     */
    public void subDivide() {
        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {

            INDArray newCorner = Nd4j.create(D);
            INDArray newWidth = Nd4j.create(D);
            for (int i = 0; i < numChildren; i++) {
                int div = 1;
                for (int d = 0; d < D; d++) {
                    newWidth.putScalar(d, .5 * boundary.width(d));
                    if ((i / div) % 2 == 1)
                        newCorner.putScalar(d, boundary.corner(d) - .5 * boundary.width(d));
                    else
                        newCorner.putScalar(d, boundary.corner(d) + .5 * boundary.width(d));
                    div *= 2;
                }

                children[i] = new SpTree(this, data, newCorner, newWidth, indices);

            }

            // Move existing points to correct children
            for (int i = 0; i < size; i++) {
                boolean success = false;
                for (int j = 0; j < this.numChildren; j++)
                    if (!success)
                        success = children[j].insert(index[i]);

                index[i] = -1;
            }

            // Empty parent node
            size = 0;
            isLeaf = false;
        }
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
        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {


            // Compute distance between point and center-of-mass
            buf.assign(data.slice(pointIndex)).subi(centerOfMass);

            double D = Nd4j.getBlasWrapper().dot(buf, buf);
            // Check whether we can use this node as a "summary"
            double maxWidth = boundary.width().max(Integer.MAX_VALUE).getDouble(0);
            // Check whether we can use this node as a "summary"
            if (isLeaf() || maxWidth / Math.sqrt(D) < theta) {

                // Compute and add t-SNE force between point and current node
                double Q = 1.0 / (1.0 + D);
                double mult = cumSize * Q;
                sumQ.addAndGet(mult);
                mult *= Q;
                negativeForce.addi(buf.muli(mult));

            } else {

                // Recursively apply Barnes-Hut to children
                for (int i = 0; i < numChildren; i++) {
                    children[i].computeNonEdgeForces(pointIndex, theta, negativeForce, sumQ);
                }

            }
        }
    }


    /**
     *
     * Compute edge forces using barnes hut
     * @param rowP a vector
     * @param colP
     * @param valP
     * @param N the number of elements
     * @param posF the positive force
     */
    public void computeEdgeForces(INDArray rowP, INDArray colP, INDArray valP, int N, INDArray posF) {
        if (!rowP.isVector())
            throw new IllegalArgumentException("RowP must be a vector");
        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);

        // Loop over all edges in the graph
        double D;
        for (int n = 0; n < N; n++) {
            INDArray slice = data.slice(n);
            for (int i = rowP.getInt(n); i < rowP.getInt(n + 1); i++) {

                // Compute pairwise distance and Q-value
                buf.assign(slice).subi(data.slice(colP.getInt(i)));

                D = 1e-12 + Nd4j.getBlasWrapper().dot(buf, buf);
                D = valP.getDouble(i) / D;

                // Sum positive force
                posF.slice(n).addi(buf.muli(D));

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
        MemoryWorkspace workspace =
                workspaceMode == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal,
                        workspaceExternal);
        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {

            for (int n = 0; n < size; n++) {
                INDArray point = data.slice(index[n]);
                if (!boundary.contains(point))
                    return false;
            }
            if (!isLeaf()) {
                boolean correct = true;
                for (int i = 0; i < numChildren; i++)
                    correct = correct && children[i].isCorrect();
                return correct;
            }

            return true;
        }
    }

    /**
     * The depth of the node
     * @return the depth of the node
     */
    public int depth() {
        if (isLeaf())
            return 1;
        int depth = 1;
        int maxChildDepth = 0;
        for (int i = 0; i < numChildren; i++) {
            maxChildDepth = Math.max(maxChildDepth, children[0].depth());
        }

        return depth + maxChildDepth;
    }

    private void fill(int n) {
        if (indices.isEmpty() && parent == null)
            for (int i = 0; i < n; i++) {
                log.trace("Inserted " + i);
                insert(i);
            }
        else
            log.warn("Called fill already");
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
