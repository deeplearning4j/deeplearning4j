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

package org.deeplearning4j.clustering.info;

import lombok.Data;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 */
@Data
public class ClusterInfo implements Serializable {

    private double averagePointDistanceFromCenter;
    private double maxPointDistanceFromCenter;
    private double pointDistanceFromCenterVariance;
    private double totalPointDistanceFromCenter;
    private boolean inverse;
    private Map<String, Double> pointDistancesFromCenter = new ConcurrentHashMap<>();

    public ClusterInfo(boolean inverse) {
        this(false, inverse);
    }

    /**
     *
     * @param threadSafe
     */
    public ClusterInfo(boolean threadSafe, boolean inverse) {
        super();
        this.inverse = inverse;
        if (threadSafe) {
            pointDistancesFromCenter = Collections.synchronizedMap(pointDistancesFromCenter);
        }
    }

    /**
     *
     * @return
     */
    public Set<Map.Entry<String, Double>> getSortedPointDistancesFromCenter() {
        SortedSet<Map.Entry<String, Double>> sortedEntries = new TreeSet<>(new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> e1, Map.Entry<String, Double> e2) {
                int res = e1.getValue().compareTo(e2.getValue());
                return res != 0 ? res : 1;
            }
        });
        sortedEntries.addAll(pointDistancesFromCenter.entrySet());
        return sortedEntries;
    }

    /**
     *
     * @return
     */
    public Set<Map.Entry<String, Double>> getReverseSortedPointDistancesFromCenter() {
        SortedSet<Map.Entry<String, Double>> sortedEntries = new TreeSet<>(new Comparator<Map.Entry<String, Double>>() {
            @Override
            public int compare(Map.Entry<String, Double> e1, Map.Entry<String, Double> e2) {
                int res = e1.getValue().compareTo(e2.getValue());
                return -(res != 0 ? res : 1);
            }
        });
        sortedEntries.addAll(pointDistancesFromCenter.entrySet());
        return sortedEntries;
    }

    /**
     *
     * @param maxDistance
     * @return
     */
    public List<String> getPointsFartherFromCenterThan(double maxDistance) {
        Set<Map.Entry<String, Double>> sorted = getReverseSortedPointDistancesFromCenter();
        List<String> ids = new ArrayList<>();
        for (Map.Entry<String, Double> entry : sorted) {
            if (inverse && entry.getValue() < -maxDistance) {
                if (entry.getValue() < -maxDistance)
                    break;
            }

            else if (entry.getValue() > maxDistance)
                break;

            ids.add(entry.getKey());
        }
        return ids;
    }



}
