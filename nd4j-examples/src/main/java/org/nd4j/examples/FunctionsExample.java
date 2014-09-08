package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.nd4j.linalg.ops.transforms.Transforms.*;


/**
 * Created by cvn on 9/7/14.
 */
public class FunctionsExample {

    public static void main(String[] args) {

        INDArray nd2 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, new int[]{2, 6});
        INDArray ndv;

        ndv = sigmoid(nd2);
        System.out.println(ndv);

        

    }
}
