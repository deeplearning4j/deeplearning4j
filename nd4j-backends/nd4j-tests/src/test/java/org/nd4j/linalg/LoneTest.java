package org.nd4j.linalg;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * Created by agibsonccc on 4/1/16.
 */
@RunWith(Parameterized.class)
public class LoneTest extends BaseNd4jTest {
    public LoneTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testSoftmaxStability() {
        INDArray input = Nd4j.create(new double[]{ -0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04}).transpose();
        System.out.println("Input transpose " + Shape.shapeToString(input.shapeInfo()));
        INDArray output = Nd4j.create(10,1);
        System.out.println("Element wise stride of output " + output.elementWiseStride());
        Nd4j.getExecutioner().exec(new SoftMax(input, output));
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
