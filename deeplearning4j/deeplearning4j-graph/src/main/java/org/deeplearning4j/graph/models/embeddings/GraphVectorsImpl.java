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

import lombok.AllArgsConstructor;
import lombok.NoArgsConstructor;
import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.models.GraphVectors;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.Comparator;
import java.util.PriorityQueue;

/** Base implementation for GraphVectors. Used in DeepWalk, and also when loading
 * graph vectors from file.
 */
@AllArgsConstructor
@NoArgsConstructor
public class GraphVectorsImpl<V, E> implements GraphVectors<V, E> {

    protected IGraph<V, E> graph;
    protected GraphVectorLookupTable lookupTable;


    @Override
    public IGraph<V, E> getGraph() {
        return graph;
    }

    @Override
    public int numVertices() {
        return lookupTable.getNumVertices();
    }

    @Override
    public int getVectorSize() {
        return lookupTable.vectorSize();
    }

    @Override
    public INDArray getVertexVector(Vertex<V> vertex) {
        return lookupTable.getVector(vertex.vertexID());
    }

    @Override
    public INDArray getVertexVector(int vertexIdx) {
        return lookupTable.getVector(vertexIdx);
    }

    @Override
    public int[] verticesNearest(int vertexIdx, int top) {

        INDArray vec = lookupTable.getVector(vertexIdx).dup();
        double norm2 = vec.norm2Number().doubleValue();


        PriorityQueue<Pair<Double, Integer>> pq =
                        new PriorityQueue<>(lookupTable.getNumVertices(), new PairComparator());

        Level1 l1 = Nd4j.getBlasWrapper().level1();
        for (int i = 0; i < numVertices(); i++) {
            if (i == vertexIdx)
                continue;

            INDArray other = lookupTable.getVector(i);
            double cosineSim = l1.dot(vec.length(), 1.0, vec, other) / (norm2 * other.norm2Number().doubleValue());

            pq.add(new Pair<>(cosineSim, i));
        }

        int[] out = new int[top];
        for (int i = 0; i < top; i++) {
            out[i] = pq.remove().getSecond();
        }

        return out;
    }

    private static class PairComparator implements Comparator<Pair<Double, Integer>> {
        @Override
        public int compare(Pair<Double, Integer> o1, Pair<Double, Integer> o2) {
            return -Double.compare(o1.getFirst(), o2.getFirst());
        }
    }

    /**Returns the cosine similarity of the vector representations of two vertices in the graph
     * @return Cosine similarity of two vertices
     */
    @Override
    public double similarity(Vertex<V> vertex1, Vertex<V> vertex2) {
        return similarity(vertex1.vertexID(), vertex2.vertexID());
    }

    /**Returns the cosine similarity of the vector representations of two vertices in the graph,
     * given the indices of these verticies
     * @return Cosine similarity of two vertices
     */
    @Override
    public double similarity(int vertexIdx1, int vertexIdx2) {
        if (vertexIdx1 == vertexIdx2)
            return 1.0;

        INDArray vector = Transforms.unitVec(getVertexVector(vertexIdx1));
        INDArray vector2 = Transforms.unitVec(getVertexVector(vertexIdx2));
        return Nd4j.getBlasWrapper().dot(vector, vector2);
    }
}
