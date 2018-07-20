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

package org.datavec.local.transforms.misc;


import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.primitives.Pair;

/**
 * Created by Alex on 03/09/2016.
 */
public class SumLongsFunction2 implements Function<Pair<Long, Long>, Long> {
    @Override
    public Long apply(Pair<Long, Long> input) {
        return input.getFirst() + input.getSecond();
    }
}
