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

package org.datavec.local.transforms.join;

import lombok.AllArgsConstructor;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Created by huitseeker on 3/6/17. */
@AllArgsConstructor
public class ExtractKeysFunction implements Function<List<Writable>, Pair<List<Writable>, List<Writable>>> {
    private int[] columnIndexes;

    @Override
    public Pair<List<Writable>, List<Writable>> apply(List<Writable> writables) {

        List<Writable> keyValues;
        if (columnIndexes.length == 1) {
            keyValues = Collections.singletonList(writables.get(columnIndexes[0]));
        } else {
            keyValues = new ArrayList<>(columnIndexes.length);
            for (int i : columnIndexes) {
                keyValues.add(writables.get(i));
            }
        }

        return Pair.of(keyValues, writables);
    }
}
