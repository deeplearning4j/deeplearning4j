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

package org.datavec.api.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

import java.util.*;

/**
 * Column analysis class for NDArray columns
 *
 * @author Alex Black
 */
@AllArgsConstructor
@NoArgsConstructor //For Jackson/json
@Builder(builderClassName = "Builder", builderMethodName = "Builder")
@Data
public class NDArrayAnalysis implements ColumnAnalysis {

    private long countTotal;
    private long countNull;
    private long minLength;
    private long maxLength;
    private long totalNDArrayValues;
    private Map<Integer, Long> countsByRank;
    private double minValue;
    private double maxValue;
    protected double[] histogramBuckets;
    protected long[] histogramBucketCounts;


    @Override
    public ColumnType getColumnType() {
        return ColumnType.NDArray;
    }

    @Override
    public String toString() {
        Map<Integer, Long> sortedCountsByRank = new LinkedHashMap<>();
        List<Integer> keys =
                        new ArrayList<>(countsByRank == null ? Collections.<Integer>emptySet() : countsByRank.keySet());
        Collections.sort(keys);
        for (Integer i : keys) {
            sortedCountsByRank.put(i, countsByRank.get(i));
        }

        return "NDArrayAnalysis(countTotal=" + countTotal + ",countNull=" + countNull + ",minLength=" + minLength
                        + ",maxLength=" + maxLength + ",totalValuesAllNDArrays=" + totalNDArrayValues + ",minValue="
                        + minValue + ",maxValue=" + maxValue + ",countsByNDArrayRank=" + sortedCountsByRank + ")";
    }


}
