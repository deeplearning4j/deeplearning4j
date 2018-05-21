package timeseries.util;

import org.datavec.api.timeseries.util.TimeSeriesWritableUtils;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Writable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;

public class TimeSeriesUtilsTest {

    @Test
    public void testTimeSeriesCreation() {
        List<List<List<Writable>>> test = new ArrayList<>();
        List<List<Writable>> timeStep = new ArrayList<>();
        for(int i = 0; i < 5; i++) {
            timeStep.add(getRecord(5));
        }

        test.add(timeStep);

        INDArray arr = TimeSeriesWritableUtils.convertWritablesSequence(test).getFirst();
        assertArrayEquals(new long[]{1,5,5},arr.shape());
     }

     private List<Writable> getRecord(int length) {
        List<Writable> ret = new ArrayList<>();
        for(int i = 0; i < length; i++) {
            ret.add(new DoubleWritable(1.0));
        }

        return ret;
     }

}
