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

package org.deeplearning4j.rl4j.observation.transform;

import org.deeplearning4j.rl4j.observation.Observation;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.datavec.api.transform.Operation;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class TransformProcessTest {
    @Test(expected = IllegalArgumentException.class)
    public void when_noChannelNameIsSuppliedToBuild_expect_exception() {
        // Arrange
        TransformProcess.builder().build();
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_callingTransformWithNullArg_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .build("test");

        // Act
        sut.transform(null, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_callingTransformWithEmptyChannelData_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>();

        // Act
        sut.transform(channelsData, 0, false);
    }

    @Test(expected = NullPointerException.class)
    public void when_addingNullFilter_expect_nullException() {
        // Act
        TransformProcess.builder().filter(null);
    }

    @Test
    public void when_fileteredOut_expect_skippedObservationAndFollowingOperationsSkipped() {
        // Arrange
        IntegerTransformOperationMock transformOperationMock = new IntegerTransformOperationMock();
        TransformProcess sut = TransformProcess.builder()
                .filter(new FilterOperationMock(true))
                .transform("test", transformOperationMock)
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        Observation result = sut.transform(channelsData, 0, false);

        // Assert
        assertTrue(result.isSkipped());
        assertFalse(transformOperationMock.isCalled);
    }

    @Test(expected = NullPointerException.class)
    public void when_addingTransformOnNullChannel_expect_nullException() {
        // Act
        TransformProcess.builder().transform(null, new IntegerTransformOperationMock());
    }

    @Test(expected = NullPointerException.class)
    public void when_addingTransformWithNullTransform_expect_nullException() {
        // Act
        TransformProcess.builder().transform("test", null);
    }

    @Test
    public void when_transformIsCalled_expect_channelDataTransformedInSameOrder() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .filter(new FilterOperationMock(false))
                .transform("test", new IntegerTransformOperationMock())
                .transform("test", new ToDataSetTransformOperationMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        Observation result = sut.transform(channelsData, 0, false);

        // Assert
        assertFalse(result.isSkipped());
        assertEquals(-1.0, result.getData().getDouble(0), 0.00001);
    }

    @Test(expected = NullPointerException.class)
    public void when_addingPreProcessOnNullChannel_expect_nullException() {
        // Act
        TransformProcess.builder().preProcess(null, new DataSetPreProcessorMock());
    }

    @Test(expected = NullPointerException.class)
    public void when_addingPreProcessWithNullTransform_expect_nullException() {
        // Act
        TransformProcess.builder().transform("test", null);
    }

    @Test
    public void when_preProcessIsCalled_expect_channelDataPreProcessedInSameOrder() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .filter(new FilterOperationMock(false))
                .transform("test", new IntegerTransformOperationMock())
                .transform("test", new ToDataSetTransformOperationMock())
                .preProcess("test", new DataSetPreProcessorMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        Observation result = sut.transform(channelsData, 0, false);

        // Assert
        assertFalse(result.isSkipped());
        assertEquals(2, result.getData().shape().length);
        assertEquals(1, result.getData().shape()[0]);
        assertEquals(-10.0, result.getData().getDouble(0), 0.00001);
    }

    @Test(expected = IllegalStateException.class)
    public void when_transformingNullData_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .transform("test", new IntegerTransformOperationMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        Observation result = sut.transform(channelsData, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_transformingAndChannelsNotDataSet_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .preProcess("test", new DataSetPreProcessorMock())
                .build("test");

        // Act
        Observation result = sut.transform(null, 0, false);
    }


    @Test(expected = IllegalArgumentException.class)
    public void when_transformingAndChannelsEmptyDataSet_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .preProcess("test", new DataSetPreProcessorMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>();

        // Act
        Observation result = sut.transform(channelsData, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_buildIsCalledWithoutChannelNames_expect_exception() {
        // Act
        TransformProcess.builder().build();
    }

    @Test(expected = NullPointerException.class)
    public void when_buildIsCalledWithNullChannelName_expect_exception() {
        // Act
        TransformProcess.builder().build(null);
    }

    @Test
    public void when_resetIsCalled_expect_resettableAreReset() {
        // Arrange
        ResettableTransformOperationMock resettableOperation = new ResettableTransformOperationMock();
        TransformProcess sut = TransformProcess.builder()
                .filter(new FilterOperationMock(false))
                .transform("test", new IntegerTransformOperationMock())
                .transform("test", resettableOperation)
                .build("test");

        // Act
        sut.reset();

        // Assert
        assertTrue(resettableOperation.isResetCalled);
    }

    @Test
    public void when_buildIsCalledAndAllChannelsAreDataSets_expect_observation() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .transform("test", new ToDataSetTransformOperationMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        Observation result = sut.transform(channelsData, 123, true);

        // Assert
        assertFalse(result.isSkipped());

        assertEquals(1.0, result.getData().getDouble(0), 0.00001);
    }

    @Test
    public void when_buildIsCalledAndAllChannelsAreINDArrays_expect_observation() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", Nd4j.create(new double[] { 1.0 }));
        }};

        // Act
        Observation result = sut.transform(channelsData, 123, true);

        // Assert
        assertFalse(result.isSkipped());

        assertEquals(1.0, result.getData().getDouble(0), 0.00001);
    }

    @Test(expected = IllegalStateException.class)
    public void when_buildIsCalledAndChannelsNotDataSetsOrINDArrays_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        Observation result = sut.transform(channelsData, 123, true);
    }

    @Test(expected = NullPointerException.class)
    public void when_channelDataIsNull_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .transform("test", new IntegerTransformOperationMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", null);
        }};

        // Act
        sut.transform(channelsData, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_transformAppliedOnChannelNotInMap_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .transform("test", new IntegerTransformOperationMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("not-test", 1);
        }};

        // Act
        sut.transform(channelsData, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_preProcessAppliedOnChannelNotInMap_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .preProcess("test", new DataSetPreProcessorMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("not-test", 1);
        }};

        // Act
        sut.transform(channelsData, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_buildContainsChannelNotInMap_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .transform("test", new IntegerTransformOperationMock())
                .build("not-test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        sut.transform(channelsData, 0, false);
    }

    @Test(expected = IllegalArgumentException.class)
    public void when_preProcessNotAppliedOnDataSet_expect_exception() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .preProcess("test", new DataSetPreProcessorMock())
                .build("test");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("test", 1);
        }};

        // Act
        sut.transform(channelsData, 0, false);
    }

    @Test
    public void when_transformProcessHaveMultipleChannels_expect_channelsAreCreatedInTheDefinedOrder() {
        // Arrange
        TransformProcess sut = TransformProcess.builder()
                .build("channel0", "channel1");
        Map<String, Object> channelsData = new HashMap<String, Object>() {{
            put("channel0", Nd4j.create(new double[] { 123.0 }));
            put("channel1", Nd4j.create(new double[] { 234.0 }));
        }};

        // Act
        Observation result = sut.transform(channelsData, 0, false);

        // Assert
        assertEquals(2, result.numChannels());
        assertEquals(123.0, result.getChannelData(0).getDouble(0), 0.000001);
        assertEquals(234.0, result.getChannelData(1).getDouble(0), 0.000001);
    }

    private static class FilterOperationMock implements FilterOperation {

        private final boolean skipped;

        public FilterOperationMock(boolean skipped) {
            this.skipped = skipped;
        }

        @Override
        public boolean isSkipped(Map<String, Object> channelsData, int currentObservationStep, boolean isFinalObservation) {
            return skipped;
        }
    }

    private static class IntegerTransformOperationMock implements Operation<Integer, Integer> {

        public boolean isCalled = false;

        @Override
        public Integer transform(Integer input) {
            isCalled = true;
            return -input;
        }
    }

    private static class ToDataSetTransformOperationMock implements Operation<Integer, DataSet> {

        @Override
        public DataSet transform(Integer input) {
            return new org.nd4j.linalg.dataset.DataSet(Nd4j.create(new double[] { input }), null);
        }
    }

    private static class ResettableTransformOperationMock implements Operation<Integer, Integer>, ResettableOperation {

        private boolean isResetCalled = false;

        @Override
        public Integer transform(Integer input) {
            return input * 10;
        }

        @Override
        public void reset() {
            isResetCalled = true;
        }
    }

    private static class DataSetPreProcessorMock implements DataSetPreProcessor {

        @Override
        public void preProcess(DataSet dataSet) {
            dataSet.getFeatures().muli(10.0);
        }
    }
}
