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

package org.deeplearning4j.graph.models;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.Vertex;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**Vectors for nodes in a graph.
 * Provides lookup table and convenience methods for graph vectors
 */
public interface GraphVectors<V, E> extends Serializable {

    public IGraph<V, E> getGraph();

    public int numVertices();

    public int getVectorSize();

    public INDArray getVertexVector(Vertex<V> vertex);

    public INDArray getVertexVector(int vertexIdx);

    public int[] verticesNearest(int vertexIdx, int top);

    double similarity(Vertex<V> vertex1, Vertex<V> vertex2);

    double similarity(int vertexIdx1, int vertexIdx2);

}
