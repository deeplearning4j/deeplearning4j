package org.nd4j.linalg;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import static org.junit.Assert.assertEquals;


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

    @Test
    public void testFlattenedView() {
        int rows = 8;
        int cols = 8;
        int dim2 = 4;
        int length = rows* cols;
        int length3d = rows * cols * dim2;

        INDArray first = Nd4j.linspace(1,length,length).reshape('c',rows,cols);
        INDArray second = Nd4j.create(new int[]{rows,cols},'f').assign(first);
        INDArray third = Nd4j.linspace(1,length3d,length3d).reshape('c',rows,cols,dim2);
        first.addi(0.1);
        second.addi(0.2);
        third.addi(0.3);

        first = first.get(NDArrayIndex.interval(4,8), NDArrayIndex.interval(0,2,8));
        second = second.get(NDArrayIndex.interval(3,7), NDArrayIndex.all());
        third = third.permute(0,2,1);

        INDArray cAssertion = Nd4j.create(new double[]{33.10, 35.10, 37.10, 39.10, 41.10, 43.10, 45.10, 47.10, 49.10, 51.10, 53.10, 55.10, 57.10, 59.10, 61.10, 63.10});
        INDArray fAssertion = Nd4j.create(new double[] {33.10, 41.10, 49.10, 57.10, 35.10, 43.10, 51.10, 59.10, 37.10, 45.10, 53.10, 61.10, 39.10, 47.10, 55.10, 63.10});
        assertEquals(cAssertion,Nd4j.toFlattened('c', first));
        assertEquals(fAssertion,Nd4j.toFlattened('f', first));
    }
}
