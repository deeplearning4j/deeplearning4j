/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.api.preprocessor;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;
@Tag(TagNames.NDARRAY_ETL)
@NativeTag
public class PermuteDataSetPreProcessorTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_dataSetIsNull_expect_NullPointerException(Nd4jBackend backend) {
        assertThrows(NullPointerException.class,() -> {
            // Assemble
            PermuteDataSetPreProcessor sut = new PermuteDataSetPreProcessor(PermuteDataSetPreProcessor.PermutationTypes.NCHWtoNHWC);

            // Act
            sut.preProcess(null);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_emptyDatasetInInputdataSetIsNCHW_expect_emptyDataSet(Nd4jBackend backend) {
        // Assemble
        PermuteDataSetPreProcessor sut = new PermuteDataSetPreProcessor(PermuteDataSetPreProcessor.PermutationTypes.NCHWtoNHWC);
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(ds.isEmpty());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_dataSetIsNCHW_expect_dataSetTransformedToNHWC(Nd4jBackend backend) {
        // Assemble
        int numChannels = 3;
        int height = 5;
        int width = 4;
        PermuteDataSetPreProcessor sut = new PermuteDataSetPreProcessor(PermuteDataSetPreProcessor.PermutationTypes.NCHWtoNHWC);
        INDArray input = Nd4j.create(1, numChannels, height, width);
        for(int c = 0; c < numChannels; ++c) {
            for(int h = 0; h < height; ++h) {
                for(int w = 0; w < width; ++w) {
                    input.putScalar(0, c, h, w, c*100.0 + h*10.0 + w);
                }
            }
        }
        DataSet ds = new DataSet(input, null);

        // Act
        sut.preProcess(ds);

        // Assert
        INDArray result = ds.getFeatures();
        long[] shape = result.shape();
        assertEquals(1, shape[0]);
        assertEquals(height, shape[1]);
        assertEquals(width, shape[2]);
        assertEquals(numChannels, shape[3]);

        assertEquals(0.0, result.getDouble(0, 0, 0, 0), 0.0);
        assertEquals(1.0, result.getDouble(0, 0, 1, 0), 0.0);
        assertEquals(2.0, result.getDouble(0, 0, 2, 0), 0.0);
        assertEquals(3.0, result.getDouble(0, 0, 3, 0), 0.0);

        assertEquals(110.0, result.getDouble(0, 1, 0, 1), 0.0);
        assertEquals(111.0, result.getDouble(0, 1, 1, 1), 0.0);
        assertEquals(112.0, result.getDouble(0, 1, 2, 1), 0.0);
        assertEquals(113.0, result.getDouble(0, 1, 3, 1), 0.0);

        assertEquals(210.0, result.getDouble(0, 1, 0, 2), 0.0);
        assertEquals(211.0, result.getDouble(0, 1, 1, 2), 0.0);
        assertEquals(212.0, result.getDouble(0, 1, 2, 2), 0.0);
        assertEquals(213.0, result.getDouble(0, 1, 3, 2), 0.0);

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_dataSetIsNHWC_expect_dataSetTransformedToNCHW(Nd4jBackend backend) {
        // Assemble
        int numChannels = 3;
        int height = 5;
        int width = 4;
        PermuteDataSetPreProcessor sut = new PermuteDataSetPreProcessor(PermuteDataSetPreProcessor.PermutationTypes.NHWCtoNCHW);
        INDArray input = Nd4j.create(1, height, width, numChannels);
        for(int c = 0; c < numChannels; ++c) {
            for(int h = 0; h < height; ++h) {
                for(int w = 0; w < width; ++w) {
                    input.putScalar(new int[] { 0, h, w, c }, c*100.0 + h*10.0 + w);
                }
            }
        }
        DataSet ds = new DataSet(input, null);

        // Act
        sut.preProcess(ds);

        // Assert
        INDArray result = ds.getFeatures();
        long[] shape = result.shape();
        assertEquals(1, shape[0]);
        assertEquals(numChannels, shape[1]);
        assertEquals(height, shape[2]);
        assertEquals(width, shape[3]);

        assertEquals(0.0, result.getDouble(0, 0, 0, 0), 0.0);
        assertEquals(1.0, result.getDouble(0, 0, 0, 1), 0.0);
        assertEquals(2.0, result.getDouble(0, 0, 0, 2), 0.0);
        assertEquals(3.0, result.getDouble(0, 0, 0, 3), 0.0);

        assertEquals(110.0, result.getDouble(0, 1, 1, 0), 0.0);
        assertEquals(111.0, result.getDouble(0, 1, 1, 1), 0.0);
        assertEquals(112.0, result.getDouble(0, 1, 1, 2), 0.0);
        assertEquals(113.0, result.getDouble(0, 1, 1, 3), 0.0);

        assertEquals(210.0, result.getDouble(0, 2, 1, 0), 0.0);
        assertEquals(211.0, result.getDouble(0, 2, 1, 1), 0.0);
        assertEquals(212.0, result.getDouble(0, 2, 1, 2), 0.0);
        assertEquals(213.0, result.getDouble(0, 2, 1, 3), 0.0);

    }
}
