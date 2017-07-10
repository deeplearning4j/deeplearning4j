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

package org.datavec.spark.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.analysis.AnalysisCounter;

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
