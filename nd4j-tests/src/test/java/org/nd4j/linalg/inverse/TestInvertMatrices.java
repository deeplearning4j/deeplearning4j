package org.nd4j.linalg.inverse;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.checkutil.CheckUtil;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;

/**
 * Created by agibsoncccc on 12/7/15.
 */
public class TestInvertMatrices extends BaseNd4jTest {
    public TestInvertMatrices() {
    }

    public TestInvertMatrices(Nd4jBackend backend) {
        super(backend);
    }

    public TestInvertMatrices(String name) {
        super(name);
    }

    public TestInvertMatrices(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Test
    public void testInverse() {
        RealMatrix matrix = new Array2DRowRealMatrix(new double[][]{
                {1,2},
                {3,4}
        });

        RealMatrix inverse =  MatrixUtils.inverse(matrix);
        INDArray arr =  InvertMatrix.invert(Nd4j.linspace(1, 4, 4).reshape(2, 2), false);
        for(int i = 0; i < inverse.getRowDimension(); i++) {
            for(int j = 0; j < inverse.getColumnDimension(); j++) {
                assertEquals(arr.getDouble(i,j),inverse.getEntry(i,j),1e-1);
            }
        }
    }

    @Test
    public void testInverseComparison(){

        List<Pair<INDArray,String>> list = NDArrayCreationUtil.getAllTestMatricesWithShape(10, 10, 12345);

        for( Pair<INDArray,String> p : list ){
            INDArray orig = p.getFirst();
            orig.assign(Nd4j.rand(orig.shape()));
            INDArray inverse = InvertMatrix.invert(orig, false);
            RealMatrix rm = CheckUtil.convertToApacheMatrix(orig);
            RealMatrix rmInverse = new LUDecomposition(rm).getSolver().getInverse();

            INDArray expected = CheckUtil.convertFromApacheMatrix(rmInverse);
            assertTrue(p.getSecond(),CheckUtil.checkEntries(expected,inverse,1e-3,1e-4));
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
