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

import org.datavec.api.transform.sequence.merge.SequenceMerge;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * Function for merging multiple sequences, using a {@link SequenceMerge} instance.<br>
 *
 * Typical usage:<br>
 * <pre>
 * {@code
 * JavaPairRDD<SomeKey,List<List<Writable>>> myData = ...;
 * SequenceComparator comparator = ...;
 * SequenceMergeFunction<String> sequenceMergeFunction = new SequenceMergeFunction<>(new SequenceMerge(comparator));
 * JavaRDD<List<List<Writable>>> merged = myData.groupByKey().map(sequenceMergeFunction);
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class SequenceMergeFunction<T>
                implements Function<Pair<T, Iterable<List<List<Writable>>>>, List<List<Writable>>> {

    private SequenceMerge sequenceMerge;

    public SequenceMergeFunction(SequenceMerge sequenceMerge) {
        this.sequenceMerge = sequenceMerge;
    }

    @Override
    public List<List<Writable>> apply(Pair<T, Iterable<List<List<Writable>>>> t2) {
        List<List<List<Writable>>> sequences = new ArrayList<>();
        for (List<List<Writable>> l : t2.getSecond()) {
            sequences.add(l);
        }

        return sequenceMerge.mergeSequences(sequences);
    }
}
