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

package org.deeplearning4j.spark.models.sequencevectors.functions;

import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.primitives.Counter;

/**
 * Accumulator for elements count
 *
 * @author raver119@gmail.com
 */
public class ElementsFrequenciesAccumulator implements AccumulatorParam<Counter<Long>> {
    @Override
    public Counter<Long> addAccumulator(Counter<Long> c1, Counter<Long> c2) {
        if (c1 == null) {
            return new Counter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }

    @Override
    public Counter<Long> addInPlace(Counter<Long> r1, Counter<Long> r2) {
        r1.incrementAll(r2);
        return r1;
    }

    @Override
    public Counter<Long> zero(Counter<Long> initialValue) {
        return new Counter<>();
    }
}
