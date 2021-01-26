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
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * Object dependency tracker, using object identity (not object equality) for the Ys (of type T)<br>
 * See {@link AbstractDependencyTracker} for more details
 *
 * @author Alex Black
 */
@Slf4j
public class IdentityDependencyTracker<T, D> extends AbstractDependencyTracker<T,D> {

    @Override
    protected Map<T, ?> newTMap() {
        return new IdentityHashMap<>();
    }

    @Override
    protected Set<T> newTSet() {
        return Collections.newSetFromMap(new IdentityHashMap<T, Boolean>());
    }

    @Override
    protected String toStringT(T t) {
        if(t instanceof INDArray){
            INDArray i = (INDArray)t;
            return System.identityHashCode(t) + " - id=" + i.getId() + ", " + i.shapeInfoToString();
        } else {
            return System.identityHashCode(t) + " - " + t.toString();
        }
    }

    @Override
    protected String toStringD(D d) {
        return d.toString();
    }
}
