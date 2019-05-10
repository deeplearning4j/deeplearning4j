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
public class MaxPerPartitionAccumulator implements AccumulatorParam<Counter<Integer>> {

    @Override
    public Counter<Integer> addInPlace(Counter<Integer> c1, Counter<Integer> c2) {
        c1.incrementAll(c2);
        return c1;
    }

    @Override
    public Counter<Integer> zero(Counter<Integer> initialCounter) {
        return new Counter<>();
    }

    @Override
    public Counter<Integer> addAccumulator(Counter<Integer> c1, Counter<Integer> c2) {
        if (c1 == null) {
            return new Counter<>();
        }
        addInPlace(c1, c2);
        return c1;
    }
}
