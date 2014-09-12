package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Created by agibsonccc on 9/11/14.
 */
public class TestLargeMatrices {

    private static Logger log = LoggerFactory.getLogger(TestLargeMatrices.class);


    public static void main(String[] args) {
        INDArray n = Nd4j.linspace(1,10000000,10000000);
        System.out.println("MMUL" + n.mmul(n.transpose()));

    }

}
