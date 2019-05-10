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

package org.deeplearning4j.graph.iterator.parallel;

import org.deeplearning4j.graph.iterator.GraphWalkIterator;

import java.util.List;

/**GraphWalkIteratorProvider: implementations of this interface provide a set of GraphWalkIterator objects.
 * Intended use: parallelization. One GraphWalkIterator per thread.
 */
public interface GraphWalkIteratorProvider<V> {

    /**Get a list of GraphWalkIterators. In general: may return less than the specified number of iterators,
     * (for example, for small networks) but never more than it
     * @param numIterators Number of iterators to return
     */
    List<GraphWalkIterator<V>> getGraphWalkIterators(int numIterators);

}
