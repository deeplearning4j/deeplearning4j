package org.deeplearning4j.datasets.iterator;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.tools.VariableMultiTimeseriesGenerator;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class AsyncMultiDataSetIteratorTest {

    /**
     * THIS TEST SHOULD BE ALWAYS RUN WITH DOUBLE PRECISION, WITHOUT ANY EXCLUSIONS
     *
     * @throws Exception
     */
    @Test
    public void testVariableTimeSeries1() throws Exception {
        AsyncMultiDataSetIterator amdsi = new AsyncMultiDataSetIterator(
                        new VariableMultiTimeseriesGenerator(1192, 1000, 32, 128, 10, 500, 10), 2, true);

        for (int e = 0; e < 10; e++) {
            int cnt = 0;
            while (amdsi.hasNext()) {
                MultiDataSet mds = amdsi.next();


                //log.info("Features ptr: {}", AtomicAllocator.getInstance().getPointer(mds.getFeatures()[0].data()).address());
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt,
                                mds.getFeatures()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.25,
                                mds.getLabels()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.5,
                                mds.getFeaturesMaskArrays()[0].meanNumber().doubleValue(), 1e-10);
                assertEquals("Failed on epoch " + e + "; iteration: " + cnt + ";", (double) cnt + 0.75,
                                mds.getLabelsMaskArrays()[0].meanNumber().doubleValue(), 1e-10);

                cnt++;
            }

            amdsi.reset();
            log.info("Epoch {} finished...", e);
        }
    }
}
