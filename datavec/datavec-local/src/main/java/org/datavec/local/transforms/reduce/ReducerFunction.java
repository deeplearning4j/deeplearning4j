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

package org.datavec.local.transforms.reduce;

import lombok.AllArgsConstructor;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;

import java.util.List;

/**
 * Function for executing
 * a reduction of a set of examples by key
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class ReducerFunction implements Function<Iterable<List<Writable>>, List<Writable>> {

    private final IAssociativeReducer reducer;

    @Override
    public List<Writable> apply(Iterable<List<Writable>> lists) {

        for (List<Writable> c : lists) {
            reducer.aggregableReducer().accept(c);
        }

        return reducer.aggregableReducer().get();
    }
}
