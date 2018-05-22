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

package org.datavec.local.transforms.misc;


import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

/**
 * Created by Alex on 03/09/2016.
 */
public class SumLongsFunction2 implements Function<Pair<Long, Long>, Long> {
    @Override
    public Long apply(Pair<Long, Long> input) {
        return input.getFirst() + input.getSecond();
    }
}
