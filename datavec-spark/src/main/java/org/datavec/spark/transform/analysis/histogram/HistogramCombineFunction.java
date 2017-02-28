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

package org.datavec.spark.transform.analysis.histogram;

import org.apache.spark.api.java.function.Function2;

import java.util.ArrayList;
import java.util.List;

/**
 * A combiner function used in the calculation of histograms
 *
 * @author Alex Black
 */
public class HistogramCombineFunction
                implements Function2<List<HistogramCounter>, List<HistogramCounter>, List<HistogramCounter>> {
    @Override
    public List<HistogramCounter> call(List<HistogramCounter> l1, List<HistogramCounter> l2) throws Exception {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;

        int size = l1.size();
        if (size != l2.size())
            throw new IllegalStateException("List lengths differ");

        List<HistogramCounter> out = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            HistogramCounter c1 = l1.get(i);
            HistogramCounter c2 = l2.get(i);

            //Normally shouldn't get null values here - but maybe for Bytes column, etc.
            if (c1 == null) {
                out.add(c2);
            } else if (c2 == null) {
                out.add(c1);
            } else {
                out.add(c1.merge(c2));
            }
        }
        return out;
    }
}
