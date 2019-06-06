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

package org.deeplearning4j.spark.text.accumulators;

import org.apache.spark.AccumulatorParam;
import org.nd4j.linalg.primitives.Counter;

/**
 * @author jeffreytang
 */
public class WordFreqAccumulator implements AccumulatorParam<Counter<String>> {

    @Override
    public Counter<String> addInPlace(Counter<String> c1, Counter<String> c2) {
        c1.incrementAll(c2);
        return c1;
    }

    @Override
    public Counter<String> zero(Counter<String> initialCounter) {
        return new Counter<>();
    }

    @Override
    public Counter<String> addAccumulator(Counter<String> c1, Counter<String> c2) {
        if (c1 == null) {
            return new Counter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }
}
