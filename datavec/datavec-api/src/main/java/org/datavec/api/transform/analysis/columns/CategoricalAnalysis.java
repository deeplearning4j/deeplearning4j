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

package org.datavec.api.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

import java.util.*;

/**
 * Analysis for categorical columns
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
@NoArgsConstructor //For Jackson deserialization
public class CategoricalAnalysis implements ColumnAnalysis {

    private Map<String, Long> mapOfCounts;


    @Override
    public String toString() {
        //Returning the counts from highest to lowest here, which seems like a useful default
        List<String> keys = new ArrayList<>(mapOfCounts.keySet());
        Collections.sort(keys, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return -Long.compare(mapOfCounts.get(o1), mapOfCounts.get(o2)); //Highest to lowest
            }
        });

        StringBuilder sb = new StringBuilder();
        sb.append("CategoricalAnalysis(CategoryCounts={");
        boolean first = true;
        for (String s : keys) {
            if (!first)
                sb.append(", ");
            first = false;

            sb.append(s).append("=").append(mapOfCounts.get(s));
        }
        sb.append("})");

        return sb.toString();
    }

    @Override
    public long getCountTotal() {
        Collection<Long> counts = mapOfCounts.values();
        long sum = 0;
        for (Long l : counts)
            sum += l;
        return sum;
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.Categorical;
    }
}
