package org.nd4j.serde.base64;

import org.apache.commons.net.util.Base64;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * NDArray as base 64
 *
 * @author Adam Gibson
 */
public class Nd4jBase64 {

    private Nd4jBase64() {}

    /**
     * Returns true if the base64
     * contains multiple arrays
     * This is delimited by tab
     * @param base64 the base 64 to test
     * @return true if the given base 64
     * is tab delimited or not
     */
    public static boolean isMultiple(String base64) {
        return base64.contains("\t");
    }

    /**
     * Returns a set of arrays
     * from base 64 that is tab delimited.
     * @param base64 the base 64 that's tab delimited
     * @return the set of arrays
     */
    public static INDArray[] arraysFromBase64(String base64) throws IOException {
        String[] base64Arr = base64.split("\t");
        INDArray[] ret = new INDArray[base64Arr.length];
        for (int i = 0; i < base64Arr.length; i++) {
            byte[] decode = Base64.decodeBase64(base64Arr[i]);
            ByteArrayInputStream bis = new ByteArrayInputStream(decode);
            DataInputStream dis = new DataInputStream(bis);
            INDArray predict = Nd4j.read(dis);
            ret[i] = predict;
        }
        return ret;
    }

    /**
     * Returns a tab delimited base 64
     * representation of the given arrays
     * @param arrays the arrays
     * @return
     * @throws IOException
     */
    public static String arraysToBase64(INDArray[] arrays) throws IOException {
        StringBuilder sb = new StringBuilder();
        //tab separate the outputs for de serialization
        for (INDArray outputArr : arrays) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            DataOutputStream dos = new DataOutputStream(bos);
            Nd4j.write(outputArr, dos);
            String base64 = Base64.encodeBase64String(bos.toByteArray());
            sb.append(base64);
            sb.append("\t");
        }

        return sb.toString();
    }

    /**
     * Returns an ndarray
     * as base 64
     * @param arr the array to write
     * @return the base 64 representation of the binary
     * ndarray
     * @throws IOException
     */
    public static String base64String(INDArray arr) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(arr, dos);
        String base64 = Base64.encodeBase64String(bos.toByteArray());
        return base64;
    }

    /**
     * Create an ndarray from a base 64
     * representation
     * @param base64 the base 64 to convert
     * @return the ndarray from base 64
     * @throws IOException
     */
    public static INDArray fromBase64(String base64) throws IOException {
        byte[] arr = Base64.decodeBase64(base64);
        ByteArrayInputStream bis = new ByteArrayInputStream(arr);
        DataInputStream dis = new DataInputStream(bis);
        INDArray predict = Nd4j.read(dis);
        return predict;
    }

}
