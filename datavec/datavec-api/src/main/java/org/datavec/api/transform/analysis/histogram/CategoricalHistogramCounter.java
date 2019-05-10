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

package org.datavec.api.transform.analysis.histogram;

import org.datavec.api.writable.Writable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A counter for building histograms of Categorical columns
 *
 * @author Alex Black
 */
public class CategoricalHistogramCounter implements HistogramCounter {
    private HashMap<String, Integer> counts = new HashMap<>();

    private List<String> stateNames;

    public CategoricalHistogramCounter(List<String> stateNames) {
        this.stateNames = stateNames;
    }

    @Override
    public HistogramCounter add(Writable w) {
        String value = w.toString();
        if (counts.containsKey(value))
            counts.put(value, counts.get(value) + 1);
        else
            counts.put(value, 1);
        return this;
    }

    @Override
    public HistogramCounter merge(HistogramCounter other) {
        if (!(other instanceof CategoricalHistogramCounter))
            throw new IllegalArgumentException("Input must be CategoricalHistogramCounter; got " + other);

        CategoricalHistogramCounter o = (CategoricalHistogramCounter) other;

        for (Map.Entry<String, Integer> entry : o.counts.entrySet()) {
            String key = entry.getKey();
            if (counts.containsKey(key))
                counts.put(key, counts.get(key) + entry.getValue());
            else
                counts.put(key, entry.getValue());
        }

        return this;
    }

    @Override
    public double[] getBins() {
        double[] bins = new double[stateNames.size() + 1];
        for (int i = 0; i < bins.length; i++) {
            bins[i] = i;
        }
        return bins;
    }

    @Override
    public long[] getCounts() {
        long[] ret = new long[stateNames.size()];
        int i = 0;
        for (String s : stateNames) {
            ret[i++] = counts.containsKey(s) ? counts.get(s) : 0;
        }
        return ret;
    }
}
