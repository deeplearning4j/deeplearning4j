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

package org.deeplearning4j.graph.data;

import org.deeplearning4j.graph.api.Edge;

/** EdgeLineProcessor is used during data loading from a file, where each edge is on a separate line<br>
 * Provides flexibility in loading graphs with arbitrary objects/properties that can be represented in a text format
 * Can also be used handle conversion of edges between non-numeric vertices to an appropriate numbered format
 * @param <E> type of the edge returned
 */
public interface EdgeLineProcessor<E> {

    /** Process a line of text into an edge.
     * May return null if line is not a valid edge (i.e., comment line etc)
     */
    Edge<E> processLine(String line);

}
