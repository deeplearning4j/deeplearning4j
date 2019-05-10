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

package org.datavec.local.transforms.analysis.aggregate;

import org.datavec.api.transform.analysis.AnalysisCounter;
import org.nd4j.linalg.function.BiFunction;

import java.util.ArrayList;
import java.util.List;

/**
 * Combine function used for undertaking analysis of a data set via Spark
 *
 * @author Alex Black
 */
public class AnalysisCombineFunction
                implements BiFunction<List<AnalysisCounter>, List<AnalysisCounter>, List<AnalysisCounter>> {
    @Override
    public List<AnalysisCounter> apply(List<AnalysisCounter> l1, List<AnalysisCounter> l2) {
        if (l1 == null)
            return l2;
        if (l2 == null)
            return l1;

        int size = l1.size();
        if (size != l2.size())
            throw new IllegalStateException("List lengths differ");

        List<AnalysisCounter> out = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            out.add(l1.get(i).merge(l2.get(i)));
        }
        return out;
    }
}
