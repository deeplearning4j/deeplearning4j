/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import lombok.extern.slf4j.Slf4j;

import java.util.*;

/**
 * Dependenci tracker. See {@link AbstractDependencyTracker} for details
 *
 * @param <T> For a dependency X -> Y, Y has type T
 * @param <D> For a dependency X -> Y, X has type D
 */
@Slf4j
public class DependencyTracker<T, D> extends AbstractDependencyTracker<T,D> {

    @Override
    protected Map<T, ?> newTMap() {
        return new HashMap<>();
    }

    @Override
    protected Set<T> newTSet() {
        return new HashSet<>();
    }

    @Override
    protected String toStringT(T t) {
        return t.toString();
    }

    @Override
    protected String toStringD(D d) {
        return d.toString();
    }
}
