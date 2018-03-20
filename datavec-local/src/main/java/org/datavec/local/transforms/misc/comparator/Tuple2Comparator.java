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

package org.datavec.local.transforms.misc.comparator;

import lombok.AllArgsConstructor;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.Comparator;

/**
 * Simple comparator: Compare {@code Tuple2<T,Long>} by Long value
 */
@AllArgsConstructor
public class Tuple2Comparator<T> implements Comparator<Pair<T, Long>>, Serializable {

    private final boolean ascending;

    @Override
    public int compare(Pair<T, Long> o1, Pair<T, Long> o2) {
        if (ascending)
            return Long.compare(o1.getSecond(), o2.getSecond());
        else
            return -Long.compare(o1.getSecond(), o2.getSecond());
    }
}
