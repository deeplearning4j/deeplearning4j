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

package org.datavec.api.transform.sequence.merge;

import lombok.Data;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Merge multiple sequences into one single sequence.
 * Requires a SequenceComparator to determine
 * the final ordering
 *
 * @author Alex Black
 */
@Data
public class SequenceMerge implements Serializable {

    private final SequenceComparator comparator;

    public SequenceMerge(SequenceComparator comparator) {
        this.comparator = comparator;
    }

    public List<List<Writable>> mergeSequences(List<List<List<Writable>>> multipleSequences) {

        //Approach here: append all time steps, then sort

        List<List<Writable>> out = new ArrayList<>();
        for (List<List<Writable>> sequence : multipleSequences) {
            out.addAll(sequence);
        }

        Collections.sort(out, comparator);

        return out;
    }
}
