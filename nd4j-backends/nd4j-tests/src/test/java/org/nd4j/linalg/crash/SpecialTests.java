package org.nd4j.linalg.crash;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class SpecialTests extends BaseNd4jTest {
    public SpecialTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testDimensionalThings1() {
        INDArray x = Nd4j.rand(new int[] {20, 30, 50});
        INDArray y = Nd4j.rand(x.shape());

        INDArray result = transform(x, y);
    }

    @Test
    public void testDimensionalThings2() {
        INDArray x = Nd4j.rand(new int[] {20, 30, 50});
        INDArray y = Nd4j.rand(x.shape());


        for (int i = 0; i < 1; i++) {
            int number = 5;
            int start = RandomUtils.nextInt(0, x.shape()[2] - number);

            transform(getView(x, start, 5), getView(y, start, 5));
        }
    }

    protected static INDArray getView(INDArray x, int from, int number) {
        return x.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(from, from + number));
    }

    protected static INDArray transform(INDArray a, INDArray b) {
        int nShape[] = new int[] {1, 2};
        INDArray a_reduced = a.sum(nShape);
        INDArray b_reduced = b.sum(nShape);

        //log.info("reduced shape: {}", Arrays.toString(a_reduced.shapeInfoDataBuffer().asInt()));

        return Transforms.abs(a_reduced.sub(b_reduced)).div(a_reduced);
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
