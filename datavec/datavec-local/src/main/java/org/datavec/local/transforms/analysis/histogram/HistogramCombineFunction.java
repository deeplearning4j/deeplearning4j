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

package org.datavec.local.transforms.analysis.histogram;

import org.datavec.api.transform.analysis.histogram.HistogramCounter;
import org.nd4j.linalg.function.BiFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * A combiner function used in the calculation of histograms
 *
 * @author Alex Black
 */
public class HistogramCombineFunction
                implements BiFunction<List<HistogramCounter>, List<HistogramCounter>, List<HistogramCounter>> {
    @Override
    public List<HistogramCounter> apply(List<HistogramCounter> l1, List<HistogramCounter> l2) {
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
