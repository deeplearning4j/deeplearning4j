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

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.List;

/**
 * Simple function to convert List<T extends SequenceElement> to Sequence<T>
 *
 * @author raver119@gmail.com
 */
public class ListSequenceConvertFunction<T extends SequenceElement> implements Function<List<T>, Sequence<T>> {
    @Override
    public Sequence<T> call(List<T> ts) throws Exception {
        Sequence<T> sequence = new Sequence<>(ts);
        return sequence;
    }
}
