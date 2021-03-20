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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;
@Tag(TagNames.NDARRAY_ETL)
@NativeTag
public class RGBtoGrayscaleDataSetPreProcessorTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_dataSetIsNull_expect_NullPointerException(Nd4jBackend backend) {
        assertThrows(NullPointerException.class,() -> {
            // Assemble
            RGBtoGrayscaleDataSetPreProcessor sut = new RGBtoGrayscaleDataSetPreProcessor();

            // Act
            sut.preProcess(null);
        });

    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_dataSetIsEmpty_expect_EmptyDataSet(Nd4jBackend backend) {
        // Assemble
        RGBtoGrayscaleDataSetPreProcessor sut = new RGBtoGrayscaleDataSetPreProcessor();
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(ds.isEmpty());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_colorsAreConverted_expect_grayScaleResult(Nd4jBackend backend) {
        // Assign
        int numChannels = 3;
        int height = 1;
        int width = 5;

        RGBtoGrayscaleDataSetPreProcessor sut = new RGBtoGrayscaleDataSetPreProcessor();
        INDArray input = Nd4j.create(2, numChannels, height, width);

        // Black, Example 1
        input.putScalar(0, 0, 0, 0, 0.0 );
        input.putScalar(0, 1, 0, 0, 0.0 );
        input.putScalar(0, 2, 0, 0, 0.0 );

        // White, Example 1
        input.putScalar(0, 0, 0, 1, 255.0 );
        input.putScalar(0, 1, 0, 1, 255.0 );
        input.putScalar(0, 2, 0, 1, 255.0 );

        // Red, Example 1
        input.putScalar(0, 0, 0, 2, 255.0 );
        input.putScalar(0, 1, 0, 2, 0.0 );
        input.putScalar(0, 2, 0, 2, 0.0 );

        // Green, Example 1
        input.putScalar(0, 0, 0, 3, 0.0 );
        input.putScalar(0, 1, 0, 3, 255.0 );
        input.putScalar(0, 2, 0, 3, 0.0 );

        // Blue, Example 1
        input.putScalar(0, 0, 0, 4, 0.0 );
        input.putScalar(0, 1, 0, 4, 0.0 );
        input.putScalar(0, 2, 0, 4, 255.0 );


        // Black, Example 2
        input.putScalar(1, 0, 0, 4, 0.0 );
        input.putScalar(1, 1, 0, 4, 0.0 );
        input.putScalar(1, 2, 0, 4, 0.0 );

        // White, Example 2
        input.putScalar(1, 0, 0, 3, 255.0 );
        input.putScalar(1, 1, 0, 3, 255.0 );
        input.putScalar(1, 2, 0, 3, 255.0 );

        // Red, Example 2
        input.putScalar(1, 0, 0, 2, 255.0 );
        input.putScalar(1, 1, 0, 2, 0.0 );
        input.putScalar(1, 2, 0, 2, 0.0 );

        // Green, Example 2
        input.putScalar(1, 0, 0, 1, 0.0 );
        input.putScalar(1, 1, 0, 1, 255.0 );
        input.putScalar(1, 2, 0, 1, 0.0 );

        // Blue, Example 2
        input.putScalar(1, 0, 0, 0, 0.0 );
        input.putScalar(1, 1, 0, 0, 0.0 );
        input.putScalar(1, 2, 0, 0, 255.0 );

        DataSet ds = new DataSet(input, null);

        // Act
        sut.preProcess(ds);

        // Assert
        INDArray result = ds.getFeatures();
        long[] shape = result.shape();

        assertEquals(3, shape.length);
        assertEquals(2, shape[0]);
        assertEquals(1, shape[1]);
        assertEquals(5, shape[2]);

        assertEquals(0.0, result.getDouble(0, 0, 0), 0.05);
        assertEquals(255.0, result.getDouble(0, 0, 1), 0.05);
        assertEquals(255.0 * 0.3, result.getDouble(0, 0, 2), 0.05);
        assertEquals(255.0 * 0.59, result.getDouble(0, 0, 3), 0.05);
        assertEquals(255.0 * 0.11, result.getDouble(0, 0, 4), 0.05);

        assertEquals(0.0, result.getDouble(1, 0, 4), 0.05);
        assertEquals(255.0, result.getDouble(1, 0, 3), 0.05);
        assertEquals(255.0 * 0.3, result.getDouble(1, 0, 2), 0.05);
        assertEquals(255.0 * 0.59, result.getDouble(1, 0, 1), 0.05);
        assertEquals(255.0 * 0.11, result.getDouble(1, 0, 0), 0.05);

    }
}
