/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.graph.models.embeddings;

import org.apache.commons.math3.util.FastMath;
import org.deeplearning4j.graph.models.BinaryTree;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/** A standard in-memory implementation of a lookup table for vector representations of the vertices in a graph
 * @author Alex Black
 */
public class InMemoryGraphLookupTable implements GraphVectorLookupTable {

    protected int nVertices;
    protected int vectorSize;
    protected BinaryTree tree;
    protected INDArray vertexVectors; //'input' vectors
    protected INDArray outWeights; //'output' vectors. Specifically vectors for inner nodes in binary tree
    protected double learningRate;

    protected double[] expTable;
    protected static double MAX_EXP = 6;

    public InMemoryGraphLookupTable(int nVertices, int vectorSize, BinaryTree tree, double learningRate) {
        this.nVertices = nVertices;
        this.vectorSize = vectorSize;
        this.tree = tree;
        this.learningRate = learningRate;
        resetWeights();

        expTable = new double[1000];
        for (int i = 0; i < expTable.length; i++) {
            double tmp = FastMath.exp((i / (double) expTable.length * 2 - 1) * MAX_EXP);
            expTable[i] = tmp / (tmp + 1.0);
        }
    }

    public INDArray getVertexVectors() {
        return vertexVectors;
    }

    public INDArray getOutWeights() {
        return outWeights;
    }

    @Override
    public int vectorSize() {
        return vectorSize;
    }

    @Override
    public void resetWeights() {
        this.vertexVectors = Nd4j.rand(nVertices, vectorSize).subi(0.5).divi(vectorSize);
        this.outWeights = Nd4j.rand(nVertices - 1, vectorSize).subi(0.5).divi(vectorSize); //Full binary tree with L leaves has L-1 inner nodes
    }

    @Override
    public void iterate(int first, int second) {
        //Get vectors and gradients
        //vecAndGrads[0][0] is vector of vertex(first); vecAndGrads[1][0] is corresponding gradient
        INDArray[][] vecAndGrads = vectorsAndGradients(first, second);

        Level1 l1 = Nd4j.getBlasWrapper().level1();
        for (int i = 0; i < vecAndGrads[0].length; i++) {
            //Update: v = v - lr * gradient
            l1.axpy(vecAndGrads[0][i].length(), -learningRate, vecAndGrads[1][i], vecAndGrads[0][i]);
        }
    }

    /** Returns vertex vector and vector gradients, plus inner node vectors and inner node gradients<br>
     * Specifically, out[0] are vectors, out[1] are gradients for the corresponding vectors<br>
     * out[0][0] is vector for first vertex; out[0][1] is gradient for this vertex vector<br>
     * out[0][i] (i>0) is the inner node vector along path to second vertex; out[1][i] is gradient for inner node vertex<br>
     * This design is used primarily to aid in testing (numerical gradient checks)
     * @param first first (input) vertex index
     * @param second second (output) vertex index
     */
    public INDArray[][] vectorsAndGradients(int first, int second) {
        //Input vertex vector gradients are composed of the inner node gradients
        //Get vector for first vertex, as well as code for second:
        INDArray vec = vertexVectors.getRow(first);
        int codeLength = tree.getCodeLength(second);
        long code = tree.getCode(second);
        int[] innerNodesForVertex = tree.getPathInnerNodes(second);

        INDArray[][] out = new INDArray[2][innerNodesForVertex.length + 1];

        Level1 l1 = Nd4j.getBlasWrapper().level1();
        INDArray accumError = Nd4j.create(vec.shape());
        for (int i = 0; i < codeLength; i++) {

            //Inner node:
            int innerNodeIdx = innerNodesForVertex[i];
            boolean path = getBit(code, i); //left or right?

            INDArray innerNodeVector = outWeights.getRow(innerNodeIdx);
            double sigmoidDot = sigmoid(Nd4j.getBlasWrapper().dot(innerNodeVector, vec));



            //Calculate gradient for inner node + accumulate error:
            INDArray innerNodeGrad;
            if (path) {
                innerNodeGrad = vec.mul(sigmoidDot - 1);
                l1.axpy(vec.length(), sigmoidDot - 1, innerNodeVector, accumError);
            } else {
                innerNodeGrad = vec.mul(sigmoidDot);
                l1.axpy(vec.length(), sigmoidDot, innerNodeVector, accumError);
            }

            out[0][i + 1] = innerNodeVector;
            out[1][i + 1] = innerNodeGrad;
        }

        out[0][0] = vec;
        out[1][0] = accumError;

        return out;
    }

    /** Calculate the probability of the second vertex given the first vertex
     * i.e., P(v_second | v_first)
     * @param first index of the first vertex
     * @param second index of the second vertex
     * @return probability, P(v_second | v_first)
     */
    public double calculateProb(int first, int second) {
        //Get vector for first vertex, as well as code for second:
        INDArray vec = vertexVectors.getRow(first);
        int codeLength = tree.getCodeLength(second);
        long code = tree.getCode(second);
        int[] innerNodesForVertex = tree.getPathInnerNodes(second);

        double prob = 1.0;
        for (int i = 0; i < codeLength; i++) {
            boolean path = getBit(code, i); //left or right?
            //Inner node:
            int innerNodeIdx = innerNodesForVertex[i];
            INDArray nwi = outWeights.getRow(innerNodeIdx);

            double dot = Nd4j.getBlasWrapper().dot(nwi, vec);

            //double sigmoidDot = sigmoid(dot);
            double innerProb = (path ? sigmoid(dot) : sigmoid(-dot)); //prob of going left or right at inner node
            prob *= innerProb;
        }
        return prob;
    }

    /** Calculate score. -log P(v_second | v_first) */
    public double calculateScore(int first, int second) {
        //Score is -log P(out|in)
        double prob = calculateProb(first, second);
        return -FastMath.log(prob);
    }

    public BinaryTree getTree() {
        return tree;
    }

    public INDArray getInnerNodeVector(int innerNode) {
        return outWeights.getRow(innerNode);
    }

    @Override
    public INDArray getVector(int idx) {
        return vertexVectors.getRow(idx);
    }

    @Override
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public int getNumVertices() {
        return nVertices;
    }

    private static double sigmoid(double in) {
        return 1.0 / (1.0 + FastMath.exp(-in));
    }

    private boolean getBit(long in, int bitNum) {
        long mask = 1L << bitNum;
        return (in & mask) != 0L;
    }

    public void setVertexVectors(INDArray vertexVectors) {
        this.vertexVectors = vertexVectors;
    }
}
