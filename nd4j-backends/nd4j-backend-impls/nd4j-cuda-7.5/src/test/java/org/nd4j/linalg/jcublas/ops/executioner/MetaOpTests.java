package org.nd4j.linalg.jcublas.ops.executioner;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.impl.meta.LinearMetaOp;
import org.nd4j.linalg.api.ops.impl.scalar.ScalarAdd;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class MetaOpTests {
    @Before
    public void setUp() {
        CudaEnvironment.getInstance().getConfiguration()
                .enableDebug(true);
    }


    @Test
    public void testLinearMetaOp1() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f});
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f});
        INDArray exp2 = Nd4j.create(new float[]{11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f});

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Abs opB = new Abs(array);

        LinearMetaOp metaOp = new LinearMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.TRANSFORM, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));
        assertEquals(exp, array);

        time1 = System.nanoTime();
        Nd4j.getExecutioner().exec(opA);
        Nd4j.getExecutioner().exec(opB);
        time2 = System.nanoTime();

        System.out.println("Execution time Linear: " + ((time2 - time1) / 1));

        assertEquals(exp2, array);
    }

    @Test
    public void testLinearMetaOp2() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

        INDArray array = Nd4j.create(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f});
        INDArray exp = Nd4j.create(new float[]{21f, 22f, 23f, 24f, 25f, 26f, 27f, 28f, 29f, 30f});
        INDArray exp2 = Nd4j.create(new float[]{31f, 32f, 33f, 34f, 35f, 36f, 37f, 38f, 39f, 40f});

        ScalarAdd opB = new ScalarAdd(array, 10f);

        Abs opA = new Abs(array);

        LinearMetaOp metaOp = new LinearMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.TRANSFORM, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        executioner.exec(metaOp);
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 1));
        assertEquals(exp, array);

        time1 = System.nanoTime();
        Nd4j.getExecutioner().exec(opA);
        Nd4j.getExecutioner().exec(opB);
        time2 = System.nanoTime();

        System.out.println("Execution time Linear: " + ((time2 - time1) / 1));

        assertEquals(exp2, array);
    }

    @Test
    public void testPerformance1() throws Exception {
        GridExecutioner executioner = new GridExecutioner();

/*        INDArray array = Nd4j.create(new float[]{-11f, -12f, -13f, -14f, -15f, -16f, -17f, -18f, -19f, -20f});
        INDArray exp = Nd4j.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f});
        INDArray exp2 = Nd4j.create(new float[]{11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f, 19f, 20f});
        */
        INDArray array = Nd4j.create(1024);
        INDArray exp = Nd4j.create(1024);
        INDArray exp2 = Nd4j.create(1024);

        ScalarAdd opA = new ScalarAdd(array, 10f);

        Abs opB = new Abs(array);

        LinearMetaOp metaOp = new LinearMetaOp(opA, opB);

        executioner.prepareGrid(metaOp);

        GridDescriptor descriptor = metaOp.getGridDescriptor();

        assertEquals(2, descriptor.getGridDepth());
        assertEquals(2, descriptor.getGridPointers().size());

        assertEquals(Op.Type.SCALAR, descriptor.getGridPointers().get(0).getType());
        assertEquals(Op.Type.TRANSFORM, descriptor.getGridPointers().get(1).getType());

        long time1 = System.nanoTime();
        for (int x = 0; x < 100000; x++) {
            executioner.exec(metaOp);
        }
        long time2 = System.nanoTime();

        System.out.println("Execution time Meta: " + ((time2 - time1) / 100000));
      //  assertEquals(exp, array);

        time1 = System.nanoTime();
        for (int x = 0; x < 100000; x++) {
            Nd4j.getExecutioner().exec(opA);
            Nd4j.getExecutioner().exec(opB);
        }
        time2 = System.nanoTime();

        System.out.println("Execution time Linear: " + ((time2 - time1) / 100000));


      //  assertEquals(exp2, array);

    }
}
