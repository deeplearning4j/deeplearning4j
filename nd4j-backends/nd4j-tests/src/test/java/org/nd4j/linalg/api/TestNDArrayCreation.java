package org.nd4j.linalg.api;

import org.bytedeco.javacpp.Pointer;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * Created by Alex on 30/04/2016.
 */

public class TestNDArrayCreation extends BaseNd4jTest {


    public TestNDArrayCreation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testCreateNpy() throws Exception {
        INDArray arrCreate = Nd4j.createFromNpyFile(new ClassPathResource("test.npy").getFile());
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
