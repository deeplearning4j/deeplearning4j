package org.deeplearning4j.nn;

/**
 * Weight initialization scheme
 * @author Adam Gibson
 */
public enum WeightInit {
    /*
        Variance normalized initialization (VI) (Glorot)
        Sparse initialization (SI) (Martens)
        Zeros: straight zeros
     */
    VI,SI,ZERO

}
