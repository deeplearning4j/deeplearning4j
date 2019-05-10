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

package org.deeplearning4j.nn.graph.vertex;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;

import java.io.Serializable;

/**VertexIndices defines a pair of integers: the index of a vertex, and the edge number of that vertex.
 * This is used for example in {@link org.deeplearning4j.nn.graph.ComputationGraph} to define the connection structure
 * between {@link GraphVertex} objects in the graph
 */
@AllArgsConstructor
@EqualsAndHashCode
public class VertexIndices implements Serializable {

    private final int vertexIndex;
    private final int vertexEdgeNumber;


    /**Index of the vertex */
    public int getVertexIndex() {
        return this.vertexIndex;
    }

    /** The edge number. Represents the index of the output of the vertex index, OR the index of the
     * input to the vertex, depending on the context
     */
    public int getVertexEdgeNumber() {
        return this.vertexEdgeNumber;
    }

    public String toString() {
        return "VertexIndices(vertexIndex=" + this.vertexIndex + ", vertexEdgeNumber=" + this.vertexEdgeNumber + ")";
    }
}
