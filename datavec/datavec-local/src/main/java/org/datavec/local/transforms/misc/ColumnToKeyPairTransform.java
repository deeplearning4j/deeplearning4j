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

package org.datavec.local.transforms.misc;

import lombok.AllArgsConstructor;

import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Extract out one writable, and map it to a pair with count 1.
 * Used to count the N most frequent values in a column,
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class ColumnToKeyPairTransform implements Function<List<Writable>, Pair<Writable, Long>> {
    private final int columnIndex;

    @Override
    public Pair<Writable, Long> apply(List<Writable> list) {
        return Pair.of(list.get(columnIndex), 1L);
    }
}
