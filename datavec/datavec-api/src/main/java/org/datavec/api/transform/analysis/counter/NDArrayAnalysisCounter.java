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

import org.datavec.api.transform.analysis.AnalysisCounter;
import org.datavec.api.transform.analysis.columns.NDArrayAnalysis;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A counter for performing analysis on NDArray columns
 *
 * @author Alex Black
 */
public class NDArrayAnalysisCounter implements AnalysisCounter<NDArrayAnalysisCounter> {

    private long countTotal;
    private long countNull;
    private long minLength = Long.MAX_VALUE;
    private long maxLength = -1;
    private long totalNDArrayValues;
    private Map<Integer, Long> countsByRank = new HashMap<>();
    private double minValue = Double.MAX_VALUE;
    private double maxValue = -Double.MAX_VALUE;

    @Override
    public NDArrayAnalysisCounter add(Writable writable) {
        NDArrayWritable n = (NDArrayWritable) writable;
        INDArray arr = n.get();
        countTotal++;
        if (arr == null) {
            countNull++;
        } else {
            minLength = Math.min(minLength, arr.length());
            maxLength = Math.max(maxLength, arr.length());

            int r = arr.rank();
            if (countsByRank.containsKey(arr.rank())) {
                countsByRank.put(r, countsByRank.get(r) + 1);
            } else {
                countsByRank.put(r, 1L);
            }

            totalNDArrayValues += arr.length();
            minValue = Math.min(minValue, arr.minNumber().doubleValue());
            maxValue = Math.max(maxValue, arr.maxNumber().doubleValue());
        }

        return this;
    }

    @Override
    public NDArrayAnalysisCounter merge(NDArrayAnalysisCounter other) {
        this.countTotal += other.countTotal;
        this.countNull += other.countNull;
        this.minLength = Math.min(this.minLength, other.minLength);
        this.maxLength = Math.max(this.maxLength, other.maxLength);
        this.totalNDArrayValues += other.totalNDArrayValues;
        Set<Integer> allKeys = new HashSet<>(countsByRank.keySet());
        allKeys.addAll(other.countsByRank.keySet());
        for (Integer i : allKeys) {
            long count = 0;
            if (countsByRank.containsKey(i)) {
                count += countsByRank.get(i);
            }
            if (other.countsByRank.containsKey(i)) {
                count += other.countsByRank.get(i);
            }
            countsByRank.put(i, count);
        }
        this.minValue = Math.min(this.minValue, other.minValue);
        this.maxValue = Math.max(this.maxValue, other.maxValue);

        return this;
    }

    public NDArrayAnalysis toAnalysisObject() {
        return NDArrayAnalysis.Builder().countTotal(countTotal).countNull(countNull).minLength(minLength)
                        .maxLength(maxLength).totalNDArrayValues(totalNDArrayValues).countsByRank(countsByRank)
                        .minValue(minValue).maxValue(maxValue).build();
    }
}
