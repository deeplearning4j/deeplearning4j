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

package org.datavec.api.util;


import org.nd4j.linalg.primitives.Pair;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;

/**
 * @deprecated Use {@link org.nd4j.linalg.collection.MultiDimensionalMap}
 */
@Deprecated
public class MultiDimensionalMap<K, T, V> extends org.nd4j.linalg.collection.MultiDimensionalMap<K, T, V>{
    @Deprecated
    public MultiDimensionalMap(Map<Pair<K, T>, V> backedMap) {
        super(backedMap);
    }
}
