package org.nd4j.linalg.shape.concat.padding;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

/**
 * @author Adam Gibson
 */
public class PaddingTestsC extends BaseNd4jTest {
    public PaddingTestsC() {
    }

    public PaddingTestsC(String name) {
        super(name);
    }

    public PaddingTestsC(String name, Nd4jBackend backend) {
        super(name, backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testPrepend() {
        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][]{
                {1, 1, 1, 1, 2},
                {1, 1, 1, 3, 4}
        });

        INDArray prepend = Nd4j.prepend(linspace, 3, 1.0, -1);
        assertEquals(assertion,prepend);


    }

    @Test
    public void testPaddingTensor() {
        //,1,1,1,1,2,2,0
        int kh = 1, kw = 1,  sy = 1,  sx = 1, ph = 2, pw = 2;
        INDArray linspaced = Nd4j.linspace(1,16,16).reshape(2,2,2,2);
        int n = linspaced.size(0);
        int c = linspaced.size(1);
        int h = linspaced.size(2);
        int w = linspaced.size(3);
        int outWidth = Convolution.outSize(h, kh, sy, ph, true);
        int outHeight = Convolution.outSize(w, kw, sx, pw, true);
        INDArray padded = Nd4j.pad(linspaced,new int[][]{
                {0, 0}
                , {0, 0}
                , {ph, ph + sy - 1}, {pw, pw + sx - 1}}, Nd4j.PadMode.CONSTANT);
        System.out.println(padded);
    }



    @Test
    public void testAppend() {
        INDArray linspace = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray otherAppend = Nd4j.append(linspace, 3, 1.0, -1);
        INDArray assertion = Nd4j.create(new double[][]{
                {1, 2, 1, 1, 1},
                {3, 4, 1, 1, 1}
        });

        assertEquals(assertion, otherAppend);
    }
}
