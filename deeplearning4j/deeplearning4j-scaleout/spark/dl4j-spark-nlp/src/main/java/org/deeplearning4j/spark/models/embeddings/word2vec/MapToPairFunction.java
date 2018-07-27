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

package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Map;

/**
 * @author jeffreytang
 */
public class MapToPairFunction implements Function<Map.Entry<VocabWord, INDArray>, Pair<VocabWord, INDArray>> {

    @Override
    public Pair<VocabWord, INDArray> call(Map.Entry<VocabWord, INDArray> pair) {
        return new Pair<>(pair.getKey(), pair.getValue());
    }
}
