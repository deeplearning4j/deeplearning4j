package org.deeplearning4j.rl4j.observation;

import org.deeplearning4j.rl4j.observation.channel.ChannelData;
import org.deeplearning4j.rl4j.observation.prefiltering.PreFilter;
import org.deeplearning4j.rl4j.observation.recorder.DataRecorder;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class ObservationHandlerTest {

    @Test(expected = IllegalArgumentException.class)
    public void when_buildingObservationWithoutData_expect_Exception() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();

        // Act
        sut.newObservation().build();
    }

    @Test
    public void when_buildingObservationWithRecorder_expect_dataIsRecorded() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        MockDataRecorder dataRecorderMock = new MockDataRecorder();
        sut.addDataRecorder(dataRecorderMock);

        // Act
        sut.newObservation()
                .addChannelData(new MockChannelData(123.0))
                .build();

        // Assert
        assertEquals(1, dataRecorderMock.recordedChannelDataList.size());
        List<ChannelData> channelDataList = dataRecorderMock.recordedChannelDataList.get(0);
        assertEquals(1, channelDataList.size());
        assertEquals(123.0, channelDataList.get(0).toINDArray().getDouble(0), 0.00001);
    }

    @Test
    public void when_observationIsFilteredOut_expect_skippedObservation() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        sut.addPreFilter(new MockPreFilter(false));

        // Act
        Observation result = sut.newObservation()
                .addChannelData(new MockChannelData(123.0))
                .build();

        // Assert
        assertTrue(result.isSkipped());
    }

    @Test
    public void when_dataWithDimension0Is1_expect_shapeUnchanged() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        sut.addPreFilter(new MockPreFilter(true));

        // Act
        Observation result = sut.newObservation()
                .addChannelData(new MockChannelData(123.0))
                .build();

        // Assert
        assertFalse(result.isSkipped());
        long[] expectedShape = new long[] { 1 };
        assertArrayEquals(expectedShape, result.getData().shape());
        assertEquals(123.0, result.getData().getDouble(0), 0.00001);
    }

    @Test
    public void when_dataWithDimension0IsNot1_expect_dimension0Is1() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        sut.addPreFilter(new MockPreFilter(true));

        // Act
        Observation result = sut.newObservation()
                .addChannelData(new MockChannelData(new double[] { 1.0, 2.0, 3.0, 4.0 }, new long[] { 2, 2}))
                .build();

        // Assert
        assertFalse(result.isSkipped());
        long[] expectedShape = new long[] { 1, 2, 2 };
        assertArrayEquals(expectedShape, result.getData().shape());

        assertEquals(1.0, result.getData().getDouble(0, 0, 0), 0.00001);
        assertEquals(2.0, result.getData().getDouble(0, 0, 1), 0.00001);
        assertEquals(3.0, result.getData().getDouble(0, 1, 0), 0.00001);
        assertEquals(4.0, result.getData().getDouble(0, 1, 1), 0.00001);
    }

    @Test
    public void when_preProcessorIsSet_expect_dataPreProcessed() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        sut.addPreFilter(new MockPreFilter(true));
        MockPreProcessor preProcessorMock = new MockPreProcessor();
        sut.setDataSetPreProcessor(preProcessorMock);


        // Act
        Observation result = sut.newObservation()
                .addChannelData(new MockChannelData(new double[] { 1.0, 2.0, 3.0, 4.0 }, new long[] { 2, 2}))
                .build();

        // Assert
        assertFalse(result.isSkipped());
        assertEquals(1, preProcessorMock.preProcessed.size());
        DataSet preProcessedDataSet = preProcessorMock.preProcessed.get(0);
        assertNotNull(preProcessedDataSet);
        assertFalse(preProcessedDataSet.isEmpty());

        assertEquals(1.0, preProcessedDataSet.getFeatures().getDouble(0, 0, 0), 0.00001);
        assertEquals(2.0, preProcessedDataSet.getFeatures().getDouble(0, 0, 1), 0.00001);
        assertEquals(3.0, preProcessedDataSet.getFeatures().getDouble(0, 1, 0), 0.00001);
        assertEquals(4.0, preProcessedDataSet.getFeatures().getDouble(0, 1, 1), 0.00001);
    }

    @Test
    public void when_buildingObservation_expect_observation() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        sut.addPreFilter(new MockPreFilter(true));


        // Act
        Observation result = sut.newObservation()
                .addChannelData(new MockChannelData(new double[] { 1.0, 2.0, 3.0, 4.0 }, new long[] { 2, 2}))
                .build();

        // Assert
        assertFalse(result.isSkipped());
        INDArray data = result.getData();
        assertEquals(1.0, data.getDouble(0, 0, 0), 0.00001);
        assertEquals(2.0, data.getDouble(0, 0, 1), 0.00001);
        assertEquals(3.0, data.getDouble(0, 1, 0), 0.00001);
        assertEquals(4.0, data.getDouble(0, 1, 1), 0.00001);
    }

    @Test
    public void when_observationHandlerIsReset_expect_observationStepReset() {
        // Assemble
        ObservationHandler sut = new ObservationHandler();
        MockPreFilter filter = new MockPreFilter(true);
        sut.addPreFilter(filter);

        // Act
        sut.newObservation()
                .addChannelData(new MockChannelData(1.0))
                .build();
        sut.newObservation()
                .addChannelData(new MockChannelData(2.0))
                .build();
        int stepBeforeReset = filter.callCurrentObservationStep;

        sut.reset();

        sut.newObservation()
                .addChannelData(new MockChannelData(3.0))
                .build();

        // Assert
        assertEquals(1, stepBeforeReset);
        assertEquals(0, filter.callCurrentObservationStep);
    }

    private static class MockChannelData implements ChannelData {

        private final double[] values;
        private final long[] shape;

        public MockChannelData(double value) {
            this(new double[] { value }, new long[] { 1 });
        }

        public MockChannelData(double[] values, long[] shape) {
            this.values = values;
            this.shape = shape;
        }

        @Override
        public INDArray toINDArray() {
            return Nd4j.create(values).reshape(shape);
        }
    }

    private static class MockDataRecorder implements DataRecorder {

        public ArrayList<List<ChannelData>> recordedChannelDataList = new ArrayList<List<ChannelData>>();

        @Override
        public void record(List<ChannelData> channelDataList) {
            recordedChannelDataList.add(channelDataList);
        }
    }

    private static class MockPreFilter implements PreFilter {

        private final boolean isPassing;

        private int callCurrentObservationStep = Integer.MIN_VALUE;

        public MockPreFilter(boolean isPassing) {

            this.isPassing = isPassing;
        }

        @Override
        public boolean isPassing(List<ChannelData> channelDataList, int currentObservationStep, boolean isFinalObservation) {
            this.callCurrentObservationStep = currentObservationStep;
            return isPassing;
        }
    }

    private static class MockPreProcessor implements DataSetPreProcessor {

        private List<DataSet> preProcessed = new ArrayList<DataSet>();

        @Override
        public void preProcess(DataSet dataSet) {
            preProcessed.add(dataSet);
        }
    }
}
