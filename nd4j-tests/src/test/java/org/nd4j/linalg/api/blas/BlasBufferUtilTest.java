package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * Created by agibsoncccc on 6/8/15.
 */
public class BlasBufferUtilTest extends BaseNd4jTest {

    @Test
    public void testFloat() {
        //0 offset
        INDArray test = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row1 = test.getRow(1);
        float[] dataAssertion = {2f,4f};
        float[] testData = BlasBufferUtil.getFloatData(row1);
        assertTrue(Arrays.equals(dataAssertion, testData));
    }


    @Test
    public void testSetData() {
        //offset and strided
        float[] data2 = {1f,3f};
        INDArray setData2 = Nd4j.create(2,2);
        INDArray row = setData2.getRow(1);
        BlasBufferUtil.setData(data2,row);
        assertTrue(Arrays.equals(data2,BlasBufferUtil.getFloatData(row)));
        //1 to 1 mapping
        float[] data = {1f,3f,2f,4f};
        INDArray setData = Nd4j.create(2,2);
        BlasBufferUtil.setData(data,setData);

        Nd4j.alloc = DataBuffer.AllocationMode.DIRECT;
        setData2 = Nd4j.create(2,2);
        row = setData2.getRow(1);
        BlasBufferUtil.setData(data2,row);
        assertTrue(Arrays.equals(data2, BlasBufferUtil.getFloatData(row)));

        setData = Nd4j.create(2,2);
        BlasBufferUtil.setData(data, setData);

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
