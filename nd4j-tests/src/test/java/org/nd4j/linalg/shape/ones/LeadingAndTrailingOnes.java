package org.nd4j.linalg.shape.ones;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Adam Gibson
 */
public class LeadingAndTrailingOnes extends BaseNd4jTest {

    public LeadingAndTrailingOnes() {
    }

    public LeadingAndTrailingOnes(String name) {
        super(name);
    }

    public LeadingAndTrailingOnes(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public LeadingAndTrailingOnes(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testSliceConstructor() throws Exception {
        List<INDArray> testList = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            testList.add(Nd4j.scalar(i + 1));

        INDArray test = Nd4j.create(testList, new int[]{1, testList.size()});
        INDArray expected = Nd4j.create(new float[]{1, 2, 3, 4, 5}, new int[]{5, 1, 1});
        assertEquals(expected, test);
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
