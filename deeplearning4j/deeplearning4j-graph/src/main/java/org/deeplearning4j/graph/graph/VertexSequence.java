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

package org.deeplearning4j.graph.graph;

import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.IVertexSequence;
import org.deeplearning4j.graph.api.Vertex;

import java.util.NoSuchElementException;

/**A vertex sequence represents a sequences of vertices in a graph
 * @author Alex Black
 */
public class VertexSequence<V> implements IVertexSequence<V> {
    private final IGraph<V, ?> graph;
    private int[] indices;
    private int currIdx = 0;

    public VertexSequence(IGraph<V, ?> graph, int[] indices) {
        this.graph = graph;
        this.indices = indices;
    }

    @Override
    public int sequenceLength() {
        return indices.length;
    }

    @Override
    public boolean hasNext() {
        return currIdx < indices.length;
    }

    @Override
    public Vertex<V> next() {
        if (!hasNext())
            throw new NoSuchElementException();
        return graph.getVertex(indices[currIdx++]);
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }
}
