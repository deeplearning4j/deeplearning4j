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

package org.deeplearning4j.arbiter.data;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.io.IOException;
import java.util.Map;
import java.util.Random;

/**
 *
 * MnistDataProvider - a DataProvider for the MNIST data set, with configurable number of epochs, batch size
 * and RNG seed
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor
public class MnistDataProvider implements DataProvider{

    private int numEpochs;
    private int batchSize;
    private int rngSeed;

    public MnistDataProvider(int numEpochs, int batchSize){
        this(numEpochs, batchSize, new Random().nextInt());
    }

    public MnistDataProvider(@JsonProperty("numEpochs") int numEpochs, @JsonProperty("batchSize") int batchSize,
                             @JsonProperty("rngSeed") int rngSeed) {
        this.numEpochs = numEpochs;
        this.batchSize = batchSize;
        this.rngSeed = rngSeed;
    }


    @Override
    public Object trainData(Map<String, Object> dataParameters) {
        try {
            return new MultipleEpochsIterator(numEpochs, new MnistDataSetIterator(batchSize, true, rngSeed));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Object testData(Map<String, Object> dataParameters) {
        try {
            return new MnistDataSetIterator(batchSize, false, 12345);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Class<?> getDataType() {
        return DataSetIterator.class;
    }
}
