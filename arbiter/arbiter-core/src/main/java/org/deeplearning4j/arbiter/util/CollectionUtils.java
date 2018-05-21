/*-
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4j.arbiter.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

public class CollectionUtils {

    /**
     * Count the number of unique values in a collection
     */
    public static int countUnique(Collection<?> collection) {
        HashSet<Object> set = new HashSet<>(collection);
        return set.size();
    }

    /**
     * Returns a list containing only unique values in a collection
     */
    public static <T> List<T> getUnique(Collection<T> collection) {
        HashSet<T> set = new HashSet<>();
        List<T> out = new ArrayList<>();
        for (T t : collection) {
            if (!set.contains(t)) {
                out.add(t);
                set.add(t);
            }
        }
        return out;
    }

}
