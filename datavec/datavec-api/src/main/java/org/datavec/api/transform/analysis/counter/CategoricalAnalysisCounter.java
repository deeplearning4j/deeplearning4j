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

package org.datavec.api.transform.analysis.counter;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.transform.analysis.AnalysisCounter;
import org.datavec.api.writable.Writable;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A counter function for doing analysis on Categorical columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class CategoricalAnalysisCounter implements AnalysisCounter<CategoricalAnalysisCounter> {

    private Map<String, Long> counts = new HashMap<>();
    private long countTotal = 0;



    public CategoricalAnalysisCounter() {

    }


    @Override
    public CategoricalAnalysisCounter add(Writable writable) {
        String value = writable.toString();

        long newCount = 0;
        if (counts.containsKey(value)) {
            newCount = counts.get(value);
        }
        newCount++;
        counts.put(value, newCount);

        countTotal++;

        return this;
    }

    public CategoricalAnalysisCounter merge(CategoricalAnalysisCounter other) {
        Set<String> combinedKeySet = new HashSet<>(counts.keySet());
        combinedKeySet.addAll(other.counts.keySet());

        for (String s : combinedKeySet) {
            long count = 0;
            if (counts.containsKey(s))
                count += counts.get(s);
            if (other.counts.containsKey(s))
                count += other.counts.get(s);
            counts.put(s, count);
        }

        countTotal += other.countTotal;

        return this;
    }

}
