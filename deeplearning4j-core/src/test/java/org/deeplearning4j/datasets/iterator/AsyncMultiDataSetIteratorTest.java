package org.deeplearning4j.datasets.iterator;

import org.deeplearning4j.datasets.iterator.tools.VariableTimeseriesGenerator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class AsyncMultiDataSetIteratorTest {

    /**
     * THIS TEST SHOULD BE ALWAYS RUN WITH DOUBLE PRECISION, WITHOUT ANY EXCLUSIONS
     *
     * @throws Exception
     */
    @Test
    public void testVariableTimeSeries1() throws Exception {
        AsyncMultiDataSetIterator amdsi = new AsyncMultiDataSetIterator(new VariableTimeseriesGenerator(1192, 1000, 32, 128, 100, 500, 10), 2, true);

        int cnt = 0;
        while (amdsi.hasNext()) {
            MultiDataSet mds = amdsi.next();

            assertEquals((double) cnt, mds.getFeatures()[0].meanNumber().doubleValue(), 1e-10);
            assertEquals((double) cnt + 0.25, mds.getLabels()[0].meanNumber().doubleValue(), 1e-10);
            assertEquals((double) cnt + 0.5, mds.getFeaturesMaskArrays()[0].meanNumber().doubleValue(), 1e-10);
            assertEquals((double) cnt + 0.75, mds.getLabelsMaskArrays()[0].meanNumber().doubleValue(), 1e-10);

            cnt++;
        }
    }
}
