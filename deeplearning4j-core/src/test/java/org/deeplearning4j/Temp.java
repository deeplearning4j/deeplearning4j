package org.deeplearning4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;

public class Temp {

    public static void main(String[] args){

        INDArray arr = Nd4j.create(8);
        INDArray arr2 = Nd4j.create(8);

        Broadcast.mul(arr,arr2,arr,0,2);    //No dimension "2" in rank 2 matrix

    }

}
