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

package org.deeplearning4j.graph.iterator;

import org.deeplearning4j.graph.api.IVertexSequence;

/**Interface/iterator representing a sequence of walks on a graph
 * For example, a {@code GraphWalkIterator<T>} can represesnt a set of independent random walks on a graph
 */
public interface GraphWalkIterator<T> {

    /** Length of the walks returned by next()
     * Note that a walk of length {@code i} contains {@code i+1} vertices
     */
    int walkLength();

    /**Get the next vertex sequence.
     */
    IVertexSequence<T> next();

    /** Whether the iterator has any more vertex sequences. */
    boolean hasNext();

    /** Reset the graph walk iterator. */
    void reset();
}
