package org.nd4j.linalg.api;

import org.bytedeco.javacpp.Pointer;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 30/04/2016.
 */

public class TestNDArrayCreation extends BaseNd4jTest {


    public TestNDArrayCreation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testCreateNpy() throws Exception {
        INDArray arrCreate = Nd4j.createFromNpyFile(new File("/home/agibsonccc/code/nd4j/nd4j-backends/nd4j-tests/src/test/resources/test.npy"));
        assertEquals(2,arrCreate.size(0));
        assertEquals(2,arrCreate.size(1));
        assertEquals(1.0,arrCreate.getDouble(0,0),1e-1);
        assertEquals(2.0,arrCreate.getDouble(0,1),1e-1);
        assertEquals(3.0,arrCreate.getDouble(1,0),1e-1);
        assertEquals(4.0,arrCreate.getDouble(1,1),1e-1);

    }


    @Override
    public char ordering() {
        return 'c';
    }
}
