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
import org.deeplearning4j.spark.models.sequencevectors.primitives.ExtraCounter;

/**
 * Accumulator for elements count
 *
 * @author raver119@gmail.com
 */
public class ExtraElementsFrequenciesAccumulator implements AccumulatorParam<ExtraCounter<Long>> {
    @Override
    public ExtraCounter<Long> addAccumulator(ExtraCounter<Long> c1, ExtraCounter<Long> c2) {
        if (c1 == null) {
            return new ExtraCounter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }

    @Override
    public ExtraCounter<Long> addInPlace(ExtraCounter<Long> r1, ExtraCounter<Long> r2) {
        r1.incrementAll(r2);
        return r1;
    }

    @Override
    public ExtraCounter<Long> zero(ExtraCounter<Long> initialValue) {
        return new ExtraCounter<>();
    }
}
