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

package org.nd4j.util;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class SetUtils {
    protected SetUtils() {}

    // Set specific operations

    public static <T> Set<T> intersection(Collection<T> parentCollection, Collection<T> removeFromCollection) {
        Set<T> results = new HashSet<>(parentCollection);
        results.retainAll(removeFromCollection);
        return results;
    }

    public static <T> boolean intersectionP(Set<? extends T> s1, Set<? extends T> s2) {
        for (T elt : s1) {
            if (s2.contains(elt))
                return true;
        }
        return false;
    }

    public static <T> Set<T> union(Set<? extends T> s1, Set<? extends T> s2) {
        Set<T> s3 = new HashSet<>(s1);
        s3.addAll(s2);
        return s3;
    }

    /** Return is s1 \ s2 */

    public static <T> Set<T> difference(Collection<? extends T> s1, Collection<? extends T> s2) {
        Set<T> s3 = new HashSet<>(s1);
        s3.removeAll(s2);
        return s3;
    }
}


