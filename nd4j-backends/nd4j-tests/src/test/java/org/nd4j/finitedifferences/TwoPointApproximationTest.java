package org.nd4j.finitedifferences;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.function.Function;
import org.nd4j.linalg.io.ClassPathResource;

import static org.junit.Assert.assertEquals;

public class TwoPointApproximationTest {



    @Test
    public void testLinspaceDerivative() throws Exception {
        Nd4j.create(1);
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.ANY_PANIC);
       String basePath = "/two_points_approx_deriv_numpy/";
        INDArray linspace = Nd4j.createNpyFromInputStream(new ClassPathResource(basePath + "x.npy").getInputStream());
        INDArray yLinspace = Nd4j.createNpyFromInputStream(new ClassPathResource(basePath + "y.npy").getInputStream());
        Function<INDArray,INDArray> f = new Function<INDArray, INDArray>() {
            @Override
            public INDArray apply(INDArray indArray) {
                return indArray.add(1);
            }
        };

        INDArray test = TwoPointApproximation
                .approximateDerivative(f,linspace,null,yLinspace,
                        Nd4j.create(new double[] {Float.MIN_VALUE
                                ,Float.MAX_VALUE}));

        INDArray npLoad = Nd4j.createNpyFromInputStream(new ClassPathResource(basePath + "approx_deriv_small.npy").getInputStream());
        assertEquals(npLoad,test);
        System.out.println(test);

    }
    
}
