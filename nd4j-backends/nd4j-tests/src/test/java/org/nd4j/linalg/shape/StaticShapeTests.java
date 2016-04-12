package org.nd4j.linalg.shape;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class StaticShapeTests extends BaseNd4jTest {

    public StaticShapeTests(Nd4jBackend backend) {
        super(backend);
    }


    @Test
    public void testShapeInd2Sub() {
        long normalTotal = 0;
        long n = 1000;
        for(int i = 0; i < n; i++) {
            long start = System.nanoTime();
            Shape.ind2subC(new int[]{2, 2}, 1);
            long end = System.nanoTime();
            normalTotal += Math.abs(end - start);
        }

        normalTotal /= n;
        System.out.println(normalTotal);

        System.out.println("C " + Arrays.toString(Shape.ind2subC(new int[]{2, 2}, 1)));
        System.out.println("F " + Arrays.toString(Shape.ind2sub(new int[]{2, 2}, 1)));
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
