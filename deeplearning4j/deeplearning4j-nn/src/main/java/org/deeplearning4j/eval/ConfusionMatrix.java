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

package org.deeplearning4j.eval;

import org.nd4j.shade.guava.collect.HashMultiset;
import org.nd4j.shade.guava.collect.Multiset;
import lombok.Getter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @deprecated Use {@link org.nd4j.evaluation.classification.ConfusionMatrix}
 */
@Deprecated
public class ConfusionMatrix<T extends Comparable<? super T>> extends org.nd4j.evaluation.classification.ConfusionMatrix<T> {

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ConfusionMatrix}
     */
    @Deprecated
    public ConfusionMatrix(List<T> classes) {
        super(classes);
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ConfusionMatrix}
     */
    @Deprecated
    public ConfusionMatrix() {
        super();
    }

    /**
     * @deprecated Use {@link org.nd4j.evaluation.classification.ConfusionMatrix}
     */
    @Deprecated
    public ConfusionMatrix(ConfusionMatrix<T> other) {
        super(other);
    }
}
