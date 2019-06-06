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

package org.datavec.local.transforms.rank;

import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * A simple helper function for use in executing CalculateSortedRank
 *
 * @author Alex Black
 */
public class UnzipForCalculateSortedRankFunction
                implements Function<Pair<Pair<Writable, List<Writable>>, Long>, List<Writable>> {
    @Override
    public List<Writable> apply(Pair<Pair<Writable, List<Writable>>, Long> v1) {
        List<Writable> inputWritables = new ArrayList<>(v1.getFirst().getSecond());
        inputWritables.add(new LongWritable(v1.getSecond()));
        return inputWritables;
    }
}
