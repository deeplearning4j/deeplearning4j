package org.nd4j.autodiff.listeners;

import lombok.*;

/**
 *
 * Used in SameDiff {@link Listener} instances.
 * Contains information such as the current epoch, iteration and thread
 *
 * @author Alex Black
 */
@AllArgsConstructor
@EqualsAndHashCode
@ToString
@Builder
@Setter
public class At {

    private int epoch;
    private int iteration;
    private int trainingThreadNum;
    private long javaThreadNum;
    private Operation operation;

    /**
     * @return A new instance with everything set to 0, and operation set to INFERENCE
     */
    public static At defaultAt(){
        return new At(0, 0, 0, 0, Operation.INFERENCE);
    }

    /**
     * @param op Operation
     * @return A new instance with everything set to 0, except for the specified operation
     */
    public static At defaultAt(@NonNull Operation op){
        return new At(0, 0, 0, 0, op);
    }

    /**
     * @return The current training epoch
     */
    public int epoch(){
        return epoch;
    }

    /**
     * @return The current training iteration
     */
    public int iteration(){
        return iteration;
    }

    /**
     * @return The number of the SameDiff thread
     */
    public int trainingThreadNum(){
        return trainingThreadNum;
    }

    /**
     * @return The Java/JVM thread number for training
     */
    public long javaThreadNum(){
        return javaThreadNum;
    }

    /**
     * @return The current operation
     */
    public Operation operation(){
        return operation;
    }

    /**
     * @return A copy of the current At instance
     */
    public At copy(){
        return new At(epoch, iteration, trainingThreadNum, javaThreadNum, operation);
    }

    /**
     * @param operation Operation to set in the new instance
     * @return A copy of the current instance, but with the specified operation
     */
    public At copy(Operation operation){
        return new At(epoch, iteration, trainingThreadNum, javaThreadNum, operation);
    }
}
