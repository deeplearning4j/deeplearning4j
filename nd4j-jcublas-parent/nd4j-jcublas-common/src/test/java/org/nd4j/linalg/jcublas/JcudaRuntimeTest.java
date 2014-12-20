package org.nd4j.linalg.jcublas;
import jcuda.*;
import jcuda.runtime.*;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by agibsonccc on 12/17/14.
 */
public class JcudaRuntimeTest {

    @Test
    public  void test() {
        String ldPath = System.getenv("LD_LIBRARY_PATH");
        String env = System.getProperty("java.library.path");
        Pointer pointer = new Pointer();
        JCuda.cudaMalloc(pointer, 4);
        System.out.println("Pointer: "+pointer);
        JCuda.cudaFree(pointer);
    }


}
