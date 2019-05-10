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

package org.deeplearning4j.spark.impl.common.reduce;

import org.apache.spark.api.java.function.Function2;
import scala.Tuple2;

/**
 * Add both elements of a {@code Tuple2<Integer,Double>}
 */
public class IntDoubleReduceFunction
                implements Function2<Tuple2<Integer, Double>, Tuple2<Integer, Double>, Tuple2<Integer, Double>> {
    @Override
    public Tuple2<Integer, Double> call(Tuple2<Integer, Double> f, Tuple2<Integer, Double> s) throws Exception {
        return new Tuple2<>(f._1() + s._1(), f._2() + s._2());
    }
}
