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

import org.nd4j.linalg.api.ndarray.INDArray;

/**Lookup table for vector representations of the vertices in a graph
 */
public interface GraphVectorLookupTable {

    /**The size of the vector representations
     */
    int vectorSize();

    /** Reset (randomize) the weights. */
    void resetWeights();

    /** Conduct learning given a pair of vertices (in and out) */
    void iterate(int first, int second);

    /** Get the vector for the vertex with index idx */
    public INDArray getVector(int idx);

    /** Set the learning rate */
    void setLearningRate(double learningRate);

    /** Returns the number of vertices in the graph */
    int getNumVertices();

}
