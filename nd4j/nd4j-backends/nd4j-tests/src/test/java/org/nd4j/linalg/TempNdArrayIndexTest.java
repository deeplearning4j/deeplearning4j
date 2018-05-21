package org.nd4j.linalg;

import org.junit.Ignore;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * TODO !! TO REMOVE : only for getting a better understanding of indexes !!
 */
@Ignore // temporary ignored
public class TempNdArrayIndexTest {

    public static void main(String[] args) {
        //deliberately not doing a square matrix
        INDArray cOrder = Nd4j.linspace(1, 12, 12).reshape('c', 4, 3);

        System.out.println("==========================");
        System.out.println("C order..");
        System.out.println("==========================");
        System.out.println(cOrder);
        System.out.println(cOrder.shapeInfoToString());

        System.out.println("==========================");
        System.out.println("Shape and stride of a view from the above c order array");
        System.out.println("==========================");
        INDArray cOrderView = cOrder.get(NDArrayIndex.interval(2, 4), NDArrayIndex.interval(0, 2));
        System.out.println(cOrderView);
        System.out.println("This array is a view? " + cOrderView.isView());
        System.out.println(cOrderView.shapeInfoToString());

        System.out.println("==========================");
        System.out.println("Shape and stride of a view from the above c order array");
        System.out.println("==========================");
        INDArray cOrderView2 = cOrder.get(NDArrayIndex.interval(2, 4), NDArrayIndex.interval(1, 3));
        System.out.println(cOrderView2);
        System.out.println("This array is a view? " + cOrderView2.isView());
        System.out.println(cOrderView2.shapeInfoToString());

        System.out.println("==========================");
        System.out.println("One way to build an f order array");
        System.out.println("==========================");
        INDArray fOrder = Nd4j.linspace(1, 12, 12).reshape('f', 4, 3);
        System.out.println(fOrder);
        System.out.println(fOrder.shapeInfoToString());
    }
}
