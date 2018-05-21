package org.nd4j.linalg.jcublas.util;


/**
 * @author Adam Gibson
 */
public class OpUtil {

    private OpUtil() {}

    /**
     * Return op for the given character
     * throws an @link{IllegalArgumentException}
     * for any charcter != n t or c
     * @param op the character to get the op for
     * @return the op for the given character
     */
    public static int getOp(char op) {
        /*
        op = Character.toLowerCase(op);
        switch(op) {
            case 'n': return cublasOperation.CUBLAS_OP_N;
            case 't' : return cublasOperation.CUBLAS_OP_T;
            case 'c' : return cublasOperation.CUBLAS_OP_C;
            default: throw new IllegalArgumentException("No op found");
        }
        */
        throw new UnsupportedOperationException();
    }

}
