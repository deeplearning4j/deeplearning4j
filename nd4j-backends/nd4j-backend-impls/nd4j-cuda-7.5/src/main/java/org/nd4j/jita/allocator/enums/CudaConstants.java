package org.nd4j.jita.allocator.enums;

/**
 * @author raver119@gmail.com
 */
public class CudaConstants {
    public static int cudaMemcpyHostToHost = 0;      /**< Host   -> Host */
    public static int cudaMemcpyHostToDevice = 1;      /**< Host   -> Device */
    public static int cudaMemcpyDeviceToHost = 2;      /**< Device -> Host */
    public static int cudaMemcpyDeviceToDevice = 3;      /**< Device -> Device */
    public static int cudaMemcpyDefault = 4;
}
