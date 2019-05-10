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

package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.datavec.spark.functions.FlatMapFunctionAdapter;

import java.util.Iterator;

/**
 * DoubleFlatMapFunction adapter to hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to DoubleFlatMapFunction
 *
 */
public class BaseDoubleFlatMapFunctionAdaptee<T> implements DoubleFlatMapFunction<T> {

    protected final FlatMapFunctionAdapter<T, Double> adapter;

    public BaseDoubleFlatMapFunctionAdaptee(FlatMapFunctionAdapter<T, Double> adapter) {
        this.adapter = adapter;
    }

    @Override
    public Iterator<Double> call(T t) throws Exception {
        return adapter.call(t).iterator();
    }
}
