package org.nd4j.bytebuddy.arrays.assign;

/**
 * @author Adam Gibson
 */
public class SetValueInterceptor {

    public static void setValue(int[] val,int index,int value) {
        val[index] = value;
    }

}
