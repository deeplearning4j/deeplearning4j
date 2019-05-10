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

package org.nd4j.linalg.function;

import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A simple util class for collapsing a {@link Map}
 * in to a {@link List} of {@link Pair}
 * where each item in the list will be an entry in the map
 * represented by a {@link Pair} of the original key and value type.
 *
 * @author Adam Gibson
 */
public class FunctionalUtils {


    /**
     * For each key in left and right, cogroup returns the list of values
     * as a pair for each value present in left as well as right.
     * @param left the left list of pairs to join
     * @param right the right list of pairs to join
     * @param <K> the key type
     * @param <V> the value type
     * @return a map of the list of values by key for each value in the left as well as the right
     * with each element in the pair representing the values in left and right respectively.
     */
    public static <K,V> Map<K,Pair<List<V>,List<V>>> cogroup(List<Pair<K,V>> left,List<Pair<K,V>> right) {
        Map<K,Pair<List<V>,List<V>>> ret = new HashMap<>();

        //group by key first to consolidate values
        Map<K,List<V>> leftMap = groupByKey(left);
        Map<K,List<V>> rightMap = groupByKey(right);

        /**
         * Iterate over each key in the list
         * adding values to the left items
         * as values are found in the list.
         */
        for(Map.Entry<K,List<V>> entry : leftMap.entrySet()) {
            K key = entry.getKey();
            if(!ret.containsKey(key)) {
                List<V> leftListPair = new ArrayList<>();
                List<V> rightListPair = new ArrayList<>();
                Pair<List<V>,List<V>> p = Pair.of(leftListPair,rightListPair);
                ret.put(key,p);
            }

            Pair<List<V>,List<V>> p = ret.get(key);
            p.getFirst().addAll(entry.getValue());


        }

        /**
         * Iterate over each key in the list
         * adding values to the right items
         * as values are found in the list.
         */
        for(Map.Entry<K,List<V>> entry  : rightMap.entrySet()) {
            K key = entry.getKey();
            if(!ret.containsKey(key)) {
                List<V> leftListPair = new ArrayList<>();
                List<V> rightListPair = new ArrayList<>();
                Pair<List<V>,List<V>> p = Pair.of(leftListPair,rightListPair);
                ret.put(key,p);
            }

            Pair<List<V>,List<V>> p = ret.get(key);
            p.getSecond().addAll(entry.getValue());

        }

        return ret;
    }

    /**
     * Group the input pairs by the key of each pair.
     * @param listInput the list of pairs to group
     * @param <K> the key type
     * @param <V> the value type
     * @return a map representing a grouping of the
     * keys by the given input key type and list of values
     * in the grouping.
     */
    public static <K,V> Map<K,List<V>> groupByKey(List<Pair<K,V>> listInput) {
        Map<K,List<V>> ret = new HashMap<>();
        for(Pair<K,V> pair : listInput) {
            List<V> currList = ret.get(pair.getFirst());
            if(currList == null) {
                currList = new ArrayList<>();
                ret.put(pair.getFirst(),currList);
            }

            currList.add(pair.getSecond());
        }

        return ret;
    }

    /**
     * Convert a map with a set of entries of type K for key
     * and V for value in to a list of {@link Pair}
     * @param map the map to collapse
     * @param <K> the key type
     * @param <V> the value type
     * @return the collapsed map as a {@link List}
     */
    public static <K,V> List<Pair<K,V>> mapToPair(Map<K,V> map) {
        List<Pair<K,V>> ret = new ArrayList<>(map.size());
        for(Map.Entry<K,V> entry : map.entrySet()) {
            ret.add(Pair.of(entry.getKey(),entry.getValue()));
        }

        return ret;
    }

}
