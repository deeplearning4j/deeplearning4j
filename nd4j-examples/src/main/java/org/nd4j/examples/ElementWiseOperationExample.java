package org.nd4j.examples;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by cvn on 9/6/14.
 */
public class ElementWiseOperationExample {

    public static void main(String[] args) {

        INDArray nd = Nd4j.create(new float[]{1,2,3,4},new int[]{2,2});
        System.out.println("Here's your initial matrix.");
        System.out.println(nd);
         nd.addi(1);
        System.out.println("Add 1 to each element with nd.addi(1);");
        System.out.println(nd);
         nd.muli(5);
        System.out.println("Multiply each element by 5 with nd.muli(5);");
        System.out.println(nd);
         nd.subi(3);
        System.out.println("Subtract 3 from each element with nd.subi(3);");
         System.out.println(nd);
        nd.divi(2);
        System.out.println("Divide each element by 2 with nd.divi(2);");
        System.out.println(nd);

    }

}