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

package org.datavec.local.transforms.analysis.histogram;

import lombok.AllArgsConstructor;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.histogram.*;
import org.datavec.api.transform.metadata.CategoricalMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.BiFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * An adder function used in the calculation of histograms
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class HistogramAddFunction implements BiFunction<List<HistogramCounter>, List<Writable>, List<HistogramCounter>> {
    private final int nBins;
    private final Schema schema;
    private final double[][] minsMaxes;

    @Override
    public List<HistogramCounter> apply(List<HistogramCounter> histogramCounters, List<Writable> writables) {
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
