package org.nd4j.linalg.api.blas.params;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class ParamsTestsC extends BaseNd4jTest {

    public ParamsTestsC() {
    }

    public ParamsTestsC(String name) {
        super(name);
    }

    public ParamsTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    public ParamsTestsC(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testGemm() {
        INDArray a = Nd4j.create(2, 2);
        INDArray b = Nd4j.create(2,3);
        INDArray c = Nd4j.create(2,3);
        GemmParams params = new GemmParams(a,b,c);
        assertEquals(a.columns(),params.getM());
        assertEquals(b.rows(),params.getN());
        assertEquals(b.columns(),params.getK());
        assertEquals(a.columns(),params.getLda());
        assertEquals(b.columns(),params.getLdb());
        assertEquals(a.columns(),params.getLdc());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
