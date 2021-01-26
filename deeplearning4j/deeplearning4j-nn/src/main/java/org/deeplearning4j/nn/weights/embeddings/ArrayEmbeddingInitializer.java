/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.weights.embeddings;

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Embedding layer initialization from a specified array
 *
 * @author Alex Black
 */
@EqualsAndHashCode
public class ArrayEmbeddingInitializer implements EmbeddingInitializer {

    private final INDArray embeddings;

    public ArrayEmbeddingInitializer(@NonNull INDArray embeddings) {
        Preconditions.checkState(embeddings.rank() == 2, "Embedding array must be rank 2 with shape [vocabSize, vectorSize], got array with shape %ndShape", embeddings);
        this.embeddings = embeddings;
    }

    @Override
    public void loadWeightsInto(INDArray array) {
        array.assign(embeddings);
    }

    @Override
    public long vocabSize() {
        return embeddings.size(0);
    }

    @Override
    public int vectorSize() {
        return (int)embeddings.size(1);
    }

    @Override
    public boolean jsonSerializable() {
        return false;
    }
}
