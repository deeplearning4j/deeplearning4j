/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff.internal;

import java.util.function.Predicate;

public interface IDependencyMap<T, D> {
    void clear();

    void add(T dependeeGroup, D element);

    // return Iterable for each the dependeeGroup content
    Iterable<D> getDependantsForEach(T dependeeGroup);

    Iterable<D> getDependantsForGroup(T dependeeGroup);

    boolean containsAny(T dependeeGroup);

    boolean containsAnyForGroup(T dependeeGroup);

    boolean isEmpty();

    void removeGroup(T dependeeGroup);

    Iterable<D> removeGroupReturn(T dependeeGroup);

    void removeForEach(T dependeeGroup);

    Iterable<D> removeForEachResult(T dependeeGroup);

    Iterable<D> removeGroupReturn(T dependeeGroup, Predicate<D> predicate);
}