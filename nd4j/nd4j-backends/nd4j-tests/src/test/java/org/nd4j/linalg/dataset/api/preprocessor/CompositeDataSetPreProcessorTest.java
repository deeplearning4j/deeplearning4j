/*
 *  ******************************************************************************
 *  *
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

package org.nd4j.linalg.dataset.api.preprocessor;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class CompositeDataSetPreProcessorTest extends BaseNd4jTest {

    public CompositeDataSetPreProcessorTest(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test(expected = NullPointerException.class)
    public void when_preConditionsIsNull_expect_NullPointerException() {
        // Assemble
        CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor();

        // Act
        sut.preProcess(null);

    }

    @Test
    public void when_dataSetIsEmpty_expect_emptyDataSet() {
        // Assemble
        CompositeDataSetPreProcessor sut = new CompositeDataSetPreProcessor();
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        assertTrue(ds.isEmpty());
    }

    @Test
    public void when_notStoppingOnEmptyDataSet_expect_allPreProcessorsCalled() {
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

    @Test
    public void when_stoppingOnEmptyDataSetAndFirstPreProcessorClearDS_expect_firstPreProcessorsCalled() {
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

    @Test
    public void when_stoppingOnEmptyDataSet_expect_firstPreProcessorsCalled() {
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
