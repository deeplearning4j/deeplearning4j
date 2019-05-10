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

package org.datavec.spark.transform.analysis.histogram;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function2;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.histogram.*;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.ArrayList;
import java.util.List;

/**
 * An adder function used in the calculation of histograms
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class HistogramAddFunction implements Function2<List<HistogramCounter>, List<Writable>, List<HistogramCounter>> {
    private final int nBins;
    private final Schema schema;
    private final double[][] minsMaxes;

    @Override
    public List<HistogramCounter> call(List<HistogramCounter> histogramCounters, List<Writable> writables)
                    throws Exception {
        if (histogramCounters == null) {
            histogramCounters = new ArrayList<>();
            List<ColumnType> columnTypes = schema.getColumnTypes();
            int i = 0;
            for (ColumnType ct : columnTypes) {
                switch (ct) {
                    case String:
                        histogramCounters.add(new StringHistogramCounter((int) minsMaxes[i][0], (int) minsMaxes[i][1],
                                        nBins));
                        break;
                    case Integer:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Long:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Double:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Categorical:
                        CategoricalMetaData meta = (CategoricalMetaData) schema.getMetaData(i);
                        histogramCounters.add(new CategoricalHistogramCounter(meta.getStateNames()));
                        break;
                    case Time:
                        histogramCounters.add(new DoubleHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    case Bytes:
                        histogramCounters.add(null); //TODO
                        break;
                    case NDArray:
                        histogramCounters.add(new NDArrayHistogramCounter(minsMaxes[i][0], minsMaxes[i][1], nBins));
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown column type: " + ct);
                }

                i++;
            }
        }

        int size = histogramCounters.size();
        if (size != writables.size())
            throw new IllegalStateException("Writables list and number of counters does not match (" + writables.size()
                            + " vs " + size + ")");
        for (int i = 0; i < size; i++) {
            HistogramCounter hc = histogramCounters.get(i);
            if (hc != null)
                hc.add(writables.get(i));
        }

        return histogramCounters;
    }
}
