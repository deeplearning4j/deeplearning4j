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

package org.deeplearning4j.nn.modelimport.keras.preprocessing.sequence;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

public class TimeSeriesGeneratorTest {

    @Test
    public void tsGeneratorTest() throws InvalidKerasConfigurationException {
        INDArray data = Nd4j.create(50, 10);
        INDArray targets = Nd4j.create(50, 10);


        int length = 10;
        int samplingRate = 2;
        int stride = 1;
        int startIndex = 0;
        int endIndex = 49;
        int batchSize = 1;

        boolean shuffle = false;
        boolean reverse = false;

        TimeSeriesGenerator gen = new TimeSeriesGenerator(data, targets, length,
                samplingRate, stride, startIndex, endIndex, shuffle, reverse, batchSize);

        assert gen.getLength() == length;
        assert gen.getStartIndex() == startIndex + length;
        assert gen.isReverse() == reverse;
        assert gen.isShuffle() == shuffle;
        assert gen.getEndIndex() == endIndex;
        assert gen.getBatchSize() == batchSize;
        assert gen.getSamplingRate() == samplingRate;
        assert gen.getStride() == stride;

        Pair<INDArray, INDArray> next = gen.next(0);
    }
}
