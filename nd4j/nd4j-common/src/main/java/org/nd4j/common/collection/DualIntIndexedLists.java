/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.common.collection;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class DualIntIndexedLists<T> extends ConcurrentHashMap<Integer, Map<Integer, List<T>>>  {


    public List<T> getList(int firstIndex, int secondIndex) {
        return get(firstIndex).get(secondIndex);
    }

    public void put(int firstIndex, int secondIndex, List<T> list) {
        get(firstIndex).put(secondIndex, list);
    }

    public void put(int firstIndex, int secondIndex, T element) {
        get(firstIndex).get(secondIndex).add(element);
    }

    public void put(int firstIndex, int secondIndex, List<T> list, boolean createIfAbsent) {
        if(!containsKey(firstIndex) && createIfAbsent) {
            put(firstIndex, new ConcurrentHashMap<>());
        }
        get(firstIndex).put(secondIndex, list);
    }

    public void put(int firstIndex, int secondIndex, T element, boolean createIfAbsent) {
        if(!containsKey(firstIndex) && createIfAbsent) {
            put(firstIndex, new ConcurrentHashMap<>());
        }
        get(firstIndex).get(secondIndex).add(element);
    }

    public void put(int firstIndex, int secondIndex, List<T> list, boolean createIfAbsent, boolean createIfAbsent2) {
        if(!containsKey(firstIndex) && createIfAbsent) {
            put(firstIndex, new ConcurrentHashMap<>());
        }
        if(!get(firstIndex).containsKey(secondIndex) && createIfAbsent2) {
            get(firstIndex).put(secondIndex, list);
        }
    }

    public void put(int firstIndex, int secondIndex, T element, boolean createIfAbsent, boolean createIfAbsent2) {
        if(!containsKey(firstIndex) && createIfAbsent) {
            put(firstIndex, new ConcurrentHashMap<>());
        }
        if(!get(firstIndex).containsKey(secondIndex) && createIfAbsent2) {
            get(firstIndex).put(secondIndex, new java.util.ArrayList<>());
        }
        get(firstIndex).get(secondIndex).add(element);
    }

    public void addToList(int firstIndex, int secondIndex, T element) {
        get(firstIndex).get(secondIndex).add(element);
    }

    public void addToList(int firstIndex, int secondIndex, T element, boolean createIfAbsent) {
        if(!containsKey(firstIndex) && createIfAbsent) {
            put(firstIndex, new ConcurrentHashMap<>());
        }
        if(!get(firstIndex).containsKey(secondIndex) && createIfAbsent) {
            get(firstIndex).put(secondIndex, new java.util.ArrayList<>());
        }
        get(firstIndex).get(secondIndex).add(element);
    }

    public void addToList(int firstIndex, int secondIndex, List<T> list) {
        get(firstIndex).get(secondIndex).addAll(list);
    }

    public void addToList(int firstIndex, int secondIndex, List<T> list, boolean createIfAbsent) {
        if(!containsKey(firstIndex) && createIfAbsent) {
            put(firstIndex, new ConcurrentHashMap<>());
        }
        if(!get(firstIndex).containsKey(secondIndex) && createIfAbsent) {
            get(firstIndex).put(secondIndex, new java.util.ArrayList<>());
        }
        get(firstIndex).get(secondIndex).addAll(list);
    }


}
