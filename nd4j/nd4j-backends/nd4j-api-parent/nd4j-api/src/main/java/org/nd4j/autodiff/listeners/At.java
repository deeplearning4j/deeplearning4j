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
}
