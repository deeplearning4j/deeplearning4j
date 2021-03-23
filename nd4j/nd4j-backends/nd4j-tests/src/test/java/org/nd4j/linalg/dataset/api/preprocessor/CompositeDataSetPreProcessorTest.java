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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

@Tag(TagNames.NDARRAY_ETL)
@NativeTag
public class CompositeDataSetPreProcessorTest extends BaseNd4jTestWithBackends {


    @Override
    public char ordering() {
        return 'c';
    }

     @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_preConditionsIsNull_expect_NullPointerException(Nd4jBackend backend) {
        assertThrows(NullPointerException.class,() -> {
            // Assemble
            CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor();

            // Act
            sut.preProcess(null);
        });


    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_dataSetIsEmpty_expect_emptyDataSet(Nd4jBackend backend) {
        // Assemble
        CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor();
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(ds.isEmpty());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_notStoppingOnEmptyDataSet_expect_allPreProcessorsCalled(Nd4jBackend backend) {
        // Assemble
        TestDataSetPreProcessor preProcessor1 = new TestDataSetPreProcessor(true);
        TestDataSetPreProcessor preProcessor2 = new TestDataSetPreProcessor(true);
        CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor(preProcessor1, preProcessor2);
        DataSet ds = new DataSet(Nd4j.rand(2, 2), null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(preProcessor1.hasBeenCalled);
        assertTrue(preProcessor2.hasBeenCalled);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_stoppingOnEmptyDataSetAndFirstPreProcessorClearDS_expect_firstPreProcessorsCalled(Nd4jBackend backend) {
        // Assemble
        TestDataSetPreProcessor preProcessor1 = new TestDataSetPreProcessor(true);
        TestDataSetPreProcessor preProcessor2 = new TestDataSetPreProcessor(true);
        CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor(true, preProcessor1, preProcessor2);
        DataSet ds = new DataSet(Nd4j.rand(2, 2), null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(preProcessor1.hasBeenCalled);
        assertFalse(preProcessor2.hasBeenCalled);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void when_stoppingOnEmptyDataSet_expect_firstPreProcessorsCalled(Nd4jBackend backend) {
        // Assemble
        TestDataSetPreProcessor preProcessor1 = new TestDataSetPreProcessor(false);
        TestDataSetPreProcessor preProcessor2 = new TestDataSetPreProcessor(false);
        CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor(true, preProcessor1, preProcessor2);
        DataSet ds = new DataSet(Nd4j.rand(2, 2), null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(preProcessor1.hasBeenCalled);
        assertTrue(preProcessor2.hasBeenCalled);
    }

    public static class TestDataSetPreProcessor implements DataSetPreProcessor {

        private final boolean clearDataSet;

        public boolean hasBeenCalled;

        public TestDataSetPreProcessor(boolean clearDataSet) {
            this.clearDataSet = clearDataSet;
        }

        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet dataSet) {
            hasBeenCalled = true;
            if(clearDataSet) {
                dataSet.setFeatures(null);
            }
        }
    }

}
