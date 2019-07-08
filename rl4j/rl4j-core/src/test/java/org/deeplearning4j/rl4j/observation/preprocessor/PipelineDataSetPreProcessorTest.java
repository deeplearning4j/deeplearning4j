package org.deeplearning4j.rl4j.observation.preprocessor;

import org.deeplearning4j.rl4j.observation.preprocessors.PipelineDataSetPreProcessor;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class PipelineDataSetPreProcessorTest {

    @Test(expected = NullPointerException.class)
    public void when_dataSetIsNull_expect_NullPointerException() {
        // Assemble
        PipelineDataSetPreProcessor sut = PipelineDataSetPreProcessor.builder().build();

        // Act
        sut.preProcess(null);
    }

    @Test
    public void when_dataSetIsEmpty_expect_EmptyDataSet() {
        // Assemble
        PipelineDataSetPreProcessor sut = PipelineDataSetPreProcessor.builder().build();
        DataSet ds = new DataSet(null, null);

        // Act
        sut.preProcess(ds);

        // Assert
        Assert.assertTrue(ds.isEmpty());
    }

    @Test
    public void when_dataSetBecomesEmpty_expect_stopAndReturn() {
        // Assemble
        PipelineDataSetPreProcessor sut = PipelineDataSetPreProcessor.builder()
                .addPreProcessor(new TestEmptyReturningDataSetPreProcessor())
                .addPreProcessor(new TestThrowOnEmptyDataSetPreProcessor())
                .build();
        DataSet ds = new DataSet(Nd4j.rand(2, 2), null);

        // Act
        sut.preProcess(ds);

        // Assert
        Assert.assertTrue(ds.isEmpty());
    }

    @Test
    public void when_dataSetIsNotEmpty_expect_continueProcessing() {
        // Assemble
        PipelineDataSetPreProcessor sut = PipelineDataSetPreProcessor.builder()
                .addPreProcessor(new TestThrowOnEmptyDataSetPreProcessor())
                .addPreProcessor(new TestThrowOnEmptyDataSetPreProcessor())
                .build();
        DataSet ds = new DataSet(Nd4j.rand(2, 2), null);

        // Act
        sut.preProcess(ds);

        // Assert
        Assert.assertFalse(ds.isEmpty());
    }

    public static class TestEmptyReturningDataSetPreProcessor implements DataSetPreProcessor {

        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet dataSet) {
            dataSet.setFeatures(null);
        }
    }

    public static class TestThrowOnEmptyDataSetPreProcessor implements DataSetPreProcessor {

        @Override
        public void preProcess(org.nd4j.linalg.dataset.api.DataSet dataSet) {
            dataSet.getFeatures().muli(1.0);
        }
    }

}
