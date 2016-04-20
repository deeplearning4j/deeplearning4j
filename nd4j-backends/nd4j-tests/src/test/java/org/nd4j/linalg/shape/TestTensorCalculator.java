package org.nd4j.linalg.shape;

import org.apache.commons.math3.util.Pair;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator;
import org.nd4j.linalg.api.shape.tensor.TensorCalculator1d;
import org.nd4j.linalg.api.shape.tensor.TensorCalculatorFactory;
import org.nd4j.linalg.checkutil.NDArrayCreationUtil;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
@RunWith(Parameterized.class)
public class TestTensorCalculator extends BaseNd4jTest {

    public TestTensorCalculator(Nd4jBackend backend) {
        super(backend);
    }



    @Test
    public void testTensorCalculator1d() throws Exception {
        int[][] shapes = new int[6][0];
        shapes[0] = new int[]{1,10};
        shapes[1] = new int[]{4,5};
        shapes[2] = new int[]{3,4,5};
        shapes[3] = new int[]{3,4,5,6};
        shapes[4] = new int[]{3,4,5,6,7};
        shapes[5] = new int[]{3,4,5,6,7,8};

        int seed = 123;
        for(int z = 0; z < shapes.length; z++) {
            List<Pair<INDArray, String>> list;
            switch(z){
                case 0:
                case 1:
                    list = NDArrayCreationUtil.getAllTestMatricesWithShape(shapes[z][0], shapes[z][1], seed);
                    break;
                case 2:
                    list = NDArrayCreationUtil.getAll3dTestArraysWithShape(seed, shapes[z]);
                    break;
                case 3:
                    list = NDArrayCreationUtil.getAll4dTestArraysWithShape(123, shapes[z]);
                    break;
                case 4:
                    list = NDArrayCreationUtil.getAll5dTestArraysWithShape(123, shapes[z]);
                    break;
                case 5:
                    list = NDArrayCreationUtil.getAll6dTestArraysWithShape(123, shapes[z]);
                    break;
                default:
                    throw new RuntimeException();
            }

            for (int j = 0; j < list.size(); j++) {
                Pair<INDArray, String> pair = list.get(j);
                INDArray arr = pair.getFirst();
                int[] shape = arr.shape();
                if(arr.ordering() != 'c') continue;
                for (int d = 0; d < arr.rank(); d++) {
                    int nT = arr.tensorssAlongDimension(d);
                    TensorCalculator tCalc = TensorCalculatorFactory.getTensorCalculator(arr,d);
                    assertEquals(nT, tCalc.getNumTensors());
                    for (int i = 0; i < nT; i++) {
                        System.out.println("Running: " + z + " " + j + " " + d + " " + i);
                        String msg = pair.getSecond() + " - " + i + ", d="+ d;
                        INDArray tensor = arr.tensorAlongDimension(i, d);
                        assertEquals(msg, tensor.offset(), tCalc.getOffsetForTensor(i));
                        if(shape[d] > 1) {
                            //If shape[d]==1, we might get different calculations. But we don't really care about stride then as our tensor is length 1
                            assertEquals(msg, tensor.elementWiseStride(), tCalc.getElementWiseStrideForTensor());
                        } else {
                            assertEquals(msg, 1, tCalc.getTensorLength());
                        }
                        assertEquals(msg, tensor.length(), tCalc.getTensorLength());

                        //Check shape and strides:
                        assertArrayEquals(msg, tensor.shape(), tCalc.getShape());
                        assertEquals(msg, 2, tCalc.getStride().length);
                        //1 element: stride calculation might differ, but don't matter
                        if(tensor.length() > 1)
                            assertEquals(msg, tensor.stride(1), tCalc.getStride()[1]);

                        //Check values:
                        DataBuffer db = tensor.data();
                        for( int x = 0; x < tensor.length(); x++) {
                            double dTensor = tensor.getDouble(x);
                            double dCalc = db.getDouble(x * tensor.elementWiseStride());
                            assertEquals(dTensor,dCalc,1e-1);
                        }
                    }
                }
            }
        }
    }

    @Override
    public char ordering() {
        return 'c';
    }

}
