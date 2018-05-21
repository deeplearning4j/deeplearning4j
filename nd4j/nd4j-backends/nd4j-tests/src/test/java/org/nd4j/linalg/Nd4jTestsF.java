package org.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class Nd4jTestsF extends BaseNd4jTest {

    DataBuffer.Type initialType;

    public Nd4jTestsF(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Test
    public void testConcat3D_Vstack_F() throws Exception {
        int[] shape = new int[] {1, 1000, 150};
        //INDArray cOrder =  Nd4j.rand(shape,123);


        List<INDArray> cArrays = new ArrayList<>();
        List<INDArray> fArrays = new ArrayList<>();

        for (int e = 0; e < 32; e++) {
            cArrays.add(Nd4j.create(shape, 'f').assign(e));
            //            fArrays.add(cOrder.dup('f'));
        }

        Nd4j.getExecutioner().commit();

        long time1 = System.currentTimeMillis();
        INDArray res = Nd4j.vstack(cArrays);
        long time2 = System.currentTimeMillis();

        log.info("Time spent: {} ms", time2 - time1);

        for (int e = 0; e < 32; e++) {
            INDArray tad = res.tensorAlongDimension(e, 1, 2);
            assertEquals((double) e, tad.meanNumber().doubleValue(), 1e-5);
        }
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
