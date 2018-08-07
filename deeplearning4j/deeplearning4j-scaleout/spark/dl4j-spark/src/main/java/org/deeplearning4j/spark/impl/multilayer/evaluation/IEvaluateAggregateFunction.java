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

package org.deeplearning4j.spark.impl.multilayer.evaluation;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.eval.IEvaluation;

/**
 * A simple function to merge IEvaluation instances
 *
 * @author Alex Black
 */
public class IEvaluateAggregateFunction<T extends IEvaluation> implements Function2<T[], T[], T[]> {
    @Override
    public T[] call(T[] v1, T[] v2) throws Exception {
        if (v1 == null)
            return v2;
        if (v2 == null)
            return v1;
        for (int i = 0; i < v1.length; i++) {
            v1[i].merge(v2[i]);
        }
        return v1;
    }
}
