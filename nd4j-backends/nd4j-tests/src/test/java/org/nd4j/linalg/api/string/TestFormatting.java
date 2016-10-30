package org.nd4j.linalg.api.string;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.string.NDArrayStrings;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class TestFormatting extends BaseNd4jTest {

    public TestFormatting(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testTwoByTwo() {
        INDArray arr = Nd4j.create(2, 2,2,2);
        System.out.println(new NDArrayStrings().format(arr));

    }

    @Test
    public void testNd4jArrayString(){

        //2d array
        INDArray arr = Nd4j.create(new float[]{1f,20000000f,40.838383f,3f}, new int[]{2,2});

        //default
        String expected1 = "[[         1.00,20,000,000.00],\n" +
                " [        40.84,         3.00]]";
        String serializedData1 = new NDArrayStrings(",",2).format(arr);
        Assert.assertTrue(serializedData1.equals(expected1));

        //remove commas
        String expected2 = "[[       1.00,20000000.00],\n" +
                " [      40.84,       3.00]]";
        String serializedData2 = new NDArrayStrings(",",2,"######0").format(arr);
        Assert.assertTrue(serializedData2.equals(expected2));

    }


    @Override
    public char ordering() {
        return 'f';
    }
}
