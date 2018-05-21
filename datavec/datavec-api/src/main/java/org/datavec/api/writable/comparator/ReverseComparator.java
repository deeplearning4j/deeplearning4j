/*
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.writable.comparator;

import lombok.AllArgsConstructor;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.Comparator;

@AllArgsConstructor
public class ReverseComparator<T> implements Comparator<T>, Serializable {

    private final Comparator<T> comparator;

    @Override
    public int compare(T o1, T o2) {
        return -comparator.compare(o1,o2);
    }
}
