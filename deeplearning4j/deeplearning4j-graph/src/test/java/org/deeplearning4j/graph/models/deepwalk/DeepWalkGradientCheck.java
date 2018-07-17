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

package org.deeplearning4j.graph.models.deepwalk;

import org.deeplearning4j.graph.data.GraphLoader;
import org.deeplearning4j.graph.graph.Graph;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.RandomWalkIterator;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.*;

public class DeepWalkGradientCheck {

    public static final double epsilon = 1e-8;
    public static final double MAX_REL_ERROR = 1e-3;

    @Before
    public void before() {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @Test(timeout = 10000L)
    public void checkGradients() throws IOException {

        ClassPathResource cpr = new ClassPathResource("deeplearning4j-graph/testgraph_7vertices.txt");

        Graph<String, String> graph = GraphLoader
                        .loadUndirectedGraphEdgeListFile(cpr.getTempFileFromArchive().getAbsolutePath(), 7, ",");

        int vectorSize = 5;
        int windowSize = 2;

        Nd4j.getRandom().setSeed(12345);
        DeepWalk<String, String> deepWalk = new DeepWalk.Builder<String, String>().learningRate(0.01)
                        .vectorSize(vectorSize).windowSize(windowSize).build();
        deepWalk.initialize(graph);

        for (int i = 0; i < 7; i++) {
            INDArray vector = deepWalk.getVertexVector(i);
            assertArrayEquals(new long[] {1, vectorSize}, vector.shape());
            System.out.println(Arrays.toString(vector.dup().data().asFloat()));
        }

        GraphWalkIterator<String> iter = new RandomWalkIterator<>(graph, 8);

        deepWalk.fit(iter);

        //Now, to check gradients:
        InMemoryGraphLookupTable table = (InMemoryGraphLookupTable) deepWalk.lookupTable();
        GraphHuffman tree = (GraphHuffman) table.getTree();

        //For each pair of input/output vertices: check gradients
        for (int i = 0; i < 7; i++) { //in

            //First: check probabilities p(out|in)
            double[] probs = new double[7];
            double sumProb = 0.0;
            for (int j = 0; j < 7; j++) {
                probs[j] = table.calculateProb(i, j);
                assertTrue(probs[j] >= 0.0 && probs[j] <= 1.0);
                sumProb += probs[j];
            }
            assertTrue("Output probabilities do not sum to 1.0", Math.abs(sumProb - 1.0) < 1e-5);

            for (int j = 0; j < 7; j++) { //out
                //p(j|i)

                int[] pathInnerNodes = tree.getPathInnerNodes(j);

                //Calculate gradients:
                INDArray[][] vecAndGrads = table.vectorsAndGradients(i, j);
                assertEquals(2, vecAndGrads.length);
                assertEquals(pathInnerNodes.length + 1, vecAndGrads[0].length);
                assertEquals(pathInnerNodes.length + 1, vecAndGrads[1].length);

                //Calculate gradients:
                //Two types of gradients to test:
                //(a) gradient of loss fn. wrt inner node vector representation
                //(b) gradient of loss fn. wrt vector for input word


                INDArray vertexVector = table.getVector(i);

                //Check gradients for inner nodes:
                for (int p = 0; p < pathInnerNodes.length; p++) {
                    int innerNodeIdx = pathInnerNodes[p];
                    INDArray innerNodeVector = table.getInnerNodeVector(innerNodeIdx);

                    INDArray innerNodeGrad = vecAndGrads[1][p + 1];

                    for (int v = 0; v < innerNodeVector.length(); v++) {
                        double backpropGradient = innerNodeGrad.getDouble(v);

                        double origParamValue = innerNodeVector.getDouble(v);
                        innerNodeVector.putScalar(v, origParamValue + epsilon);
                        double scorePlus = table.calculateScore(i, j);
                        innerNodeVector.putScalar(v, origParamValue - epsilon);
                        double scoreMinus = table.calculateScore(i, j);
                        innerNodeVector.putScalar(v, origParamValue); //reset param so it doesn't affect later calcs


                        double numericalGradient = (scorePlus - scoreMinus) / (2 * epsilon);

                        double relError;
                        if (backpropGradient == 0.0 && numericalGradient == 0.0)
                            relError = 0.0;
                        else {
                            relError = Math.abs(backpropGradient - numericalGradient)
                                            / (Math.abs(backpropGradient) + Math.abs(numericalGradient));
                        }

                        String msg = "innerNode grad: i=" + i + ", j=" + j + ", p=" + p + ", v=" + v + " - relError: "
                                        + relError + ", scorePlus=" + scorePlus + ", scoreMinus=" + scoreMinus
                                        + ", numGrad=" + numericalGradient + ", backpropGrad = " + backpropGradient;

                        if (relError > MAX_REL_ERROR)
                            fail(msg);
                        else
                            System.out.println(msg);
                    }
                }

                //Check gradients for input word vector:
                INDArray vectorGrad = vecAndGrads[1][0];
                assertArrayEquals(vectorGrad.shape(), vertexVector.shape());
                for (int v = 0; v < vectorGrad.length(); v++) {

                    double backpropGradient = vectorGrad.getDouble(v);

                    double origParamValue = vertexVector.getDouble(v);
                    vertexVector.putScalar(v, origParamValue + epsilon);
                    double scorePlus = table.calculateScore(i, j);
                    vertexVector.putScalar(v, origParamValue - epsilon);
                    double scoreMinus = table.calculateScore(i, j);
                    vertexVector.putScalar(v, origParamValue);

                    double numericalGradient = (scorePlus - scoreMinus) / (2 * epsilon);

                    double relError;
                    if (backpropGradient == 0.0 && numericalGradient == 0.0)
                        relError = 0.0;
                    else {
                        relError = Math.abs(backpropGradient - numericalGradient)
                                        / (Math.abs(backpropGradient) + Math.abs(numericalGradient));
                    }

                    String msg = "vector grad: i=" + i + ", j=" + j + ", v=" + v + " - relError: " + relError
                                    + ", scorePlus=" + scorePlus + ", scoreMinus=" + scoreMinus + ", numGrad="
                                    + numericalGradient + ", backpropGrad = " + backpropGradient;

                    if (relError > MAX_REL_ERROR)
                        fail(msg);
                    else
                        System.out.println(msg);
                }
                System.out.println();
            }

        }

    }



    @Test(timeout = 10000L)
    public void checkGradients2() throws IOException {

        ClassPathResource cpr = new ClassPathResource("deeplearning4j-graph/graph13.txt");

        int nVertices = 13;
        Graph<String, String> graph = GraphLoader
                        .loadUndirectedGraphEdgeListFile(cpr.getTempFileFromArchive().getAbsolutePath(), 13, ",");

        int vectorSize = 10;
        int windowSize = 3;

        Nd4j.getRandom().setSeed(12345);
        DeepWalk<String, String> deepWalk = new DeepWalk.Builder<String, String>().learningRate(0.01)
                        .vectorSize(vectorSize).windowSize(windowSize).learningRate(0.01).build();
        deepWalk.initialize(graph);

        for (int i = 0; i < nVertices; i++) {
            INDArray vector = deepWalk.getVertexVector(i);
            assertArrayEquals(new long[] {1, vectorSize}, vector.shape());
            System.out.println(Arrays.toString(vector.dup().data().asFloat()));
        }

        GraphWalkIterator<String> iter = new RandomWalkIterator<>(graph, 10);

        deepWalk.fit(iter);

        //Now, to check gradients:
        InMemoryGraphLookupTable table = (InMemoryGraphLookupTable) deepWalk.lookupTable();
        GraphHuffman tree = (GraphHuffman) table.getTree();

        //For each pair of input/output vertices: check gradients
        for (int i = 0; i < nVertices; i++) { //in

            //First: check probabilities p(out|in)
            double[] probs = new double[nVertices];
            double sumProb = 0.0;
            for (int j = 0; j < nVertices; j++) {
                probs[j] = table.calculateProb(i, j);
                assertTrue(probs[j] >= 0.0 && probs[j] <= 1.0);
                sumProb += probs[j];
            }
            assertTrue("Output probabilities do not sum to 1.0 (i=" + i + "), sum=" + sumProb,
                            Math.abs(sumProb - 1.0) < 1e-5);

            for (int j = 0; j < nVertices; j++) { //out
                //p(j|i)

                int[] pathInnerNodes = tree.getPathInnerNodes(j);

                //Calculate gradients:
                INDArray[][] vecAndGrads = table.vectorsAndGradients(i, j);
                assertEquals(2, vecAndGrads.length);
                assertEquals(pathInnerNodes.length + 1, vecAndGrads[0].length);
                assertEquals(pathInnerNodes.length + 1, vecAndGrads[1].length);

                //Calculate gradients:
                //Two types of gradients to test:
                //(a) gradient of loss fn. wrt inner node vector representation
                //(b) gradient of loss fn. wrt vector for input word


                INDArray vertexVector = table.getVector(i);

                //Check gradients for inner nodes:
                for (int p = 0; p < pathInnerNodes.length; p++) {
                    int innerNodeIdx = pathInnerNodes[p];
                    INDArray innerNodeVector = table.getInnerNodeVector(innerNodeIdx);

                    INDArray innerNodeGrad = vecAndGrads[1][p + 1];

                    for (int v = 0; v < innerNodeVector.length(); v++) {
                        double backpropGradient = innerNodeGrad.getDouble(v);

                        double origParamValue = innerNodeVector.getDouble(v);
                        innerNodeVector.putScalar(v, origParamValue + epsilon);
                        double scorePlus = table.calculateScore(i, j);
                        innerNodeVector.putScalar(v, origParamValue - epsilon);
                        double scoreMinus = table.calculateScore(i, j);
                        innerNodeVector.putScalar(v, origParamValue); //reset param so it doesn't affect later calcs


                        double numericalGradient = (scorePlus - scoreMinus) / (2 * epsilon);

                        double relError;
                        if (backpropGradient == 0.0 && numericalGradient == 0.0)
                            relError = 0.0;
                        else {
                            relError = Math.abs(backpropGradient - numericalGradient)
                                            / (Math.abs(backpropGradient) + Math.abs(numericalGradient));
                        }

                        String msg = "innerNode grad: i=" + i + ", j=" + j + ", p=" + p + ", v=" + v + " - relError: "
                                        + relError + ", scorePlus=" + scorePlus + ", scoreMinus=" + scoreMinus
                                        + ", numGrad=" + numericalGradient + ", backpropGrad = " + backpropGradient;

                        if (relError > MAX_REL_ERROR)
                            fail(msg);
                        else
                            System.out.println(msg);
                    }
                }

                //Check gradients for input word vector:
                INDArray vectorGrad = vecAndGrads[1][0];
                assertArrayEquals(vectorGrad.shape(), vertexVector.shape());
                for (int v = 0; v < vectorGrad.length(); v++) {

                    double backpropGradient = vectorGrad.getDouble(v);

                    double origParamValue = vertexVector.getDouble(v);
                    vertexVector.putScalar(v, origParamValue + epsilon);
                    double scorePlus = table.calculateScore(i, j);
                    vertexVector.putScalar(v, origParamValue - epsilon);
                    double scoreMinus = table.calculateScore(i, j);
                    vertexVector.putScalar(v, origParamValue);

                    double numericalGradient = (scorePlus - scoreMinus) / (2 * epsilon);

                    double relError;
                    if (backpropGradient == 0.0 && numericalGradient == 0.0)
                        relError = 0.0;
                    else {
                        relError = Math.abs(backpropGradient - numericalGradient)
                                        / (Math.abs(backpropGradient) + Math.abs(numericalGradient));
                    }

                    String msg = "vector grad: i=" + i + ", j=" + j + ", v=" + v + " - relError: " + relError
                                    + ", scorePlus=" + scorePlus + ", scoreMinus=" + scoreMinus + ", numGrad="
                                    + numericalGradient + ", backpropGrad = " + backpropGradient;

                    if (relError > MAX_REL_ERROR)
                        fail(msg);
                    else
                        System.out.println(msg);
                }
                System.out.println();
            }

        }

    }

    private static boolean getBit(long in, int bitNum) {
        long mask = 1L << bitNum;
        return (in & mask) != 0L;
    }
}
