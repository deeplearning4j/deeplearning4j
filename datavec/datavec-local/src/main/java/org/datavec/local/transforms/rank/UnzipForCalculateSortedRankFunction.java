/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

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
