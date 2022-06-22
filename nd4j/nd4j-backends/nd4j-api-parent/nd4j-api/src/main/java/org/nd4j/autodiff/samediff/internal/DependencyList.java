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

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.nd4j.common.primitives.Pair;

@Data
@AllArgsConstructor
public class DependencyList<T, D> {
    private T dependencyFor;
    private Iterable<D> dependencies;
    private Iterable<Pair<D, D>> orDependencies;

    public Collection<D> getDependenciesAsCollection() {
        List<D> result = new ArrayList<>();
        if (dependencies != null)
            dependencies.forEach(result::add);
        return result;
    }

    public Collection<Pair<D, D>> getOrDependenciesAsCollection() {
        List<Pair<D, D>> result = new ArrayList<>();
        if (orDependencies != null)
            orDependencies.forEach(result::add);
        return result;
    }
}
