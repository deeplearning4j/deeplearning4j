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

package org.deeplearning4j.graph.api;

import java.util.Iterator;

/**Represents a sequence of vertices in a graph.<br>
 * General-purpose, but can be used to represent a walk on a graph, for example.
 */
public interface IVertexSequence<T> extends Iterator<Vertex<T>> {

    /** Length of the vertex sequence */
    int sequenceLength();

}
