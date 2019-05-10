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

package org.datavec.api.records;

/**
 * Interface that acts as an iterator for deserializing maps.
 * The deserializer returns an instance that the record uses to
 * read vectors and maps. An example of usage is as follows:
 *
 * <code>
 * Index idx = startVector(...);
 * while (!idx.done()) {
 *   .... // read element of a vector
 *   idx.incr();
 * }
 * </code>
 */
public interface Index {
    boolean done();

    void incr();
}
