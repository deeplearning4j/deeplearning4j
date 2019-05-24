package org.nd4j.autodiff.listeners;

import lombok.*;

@AllArgsConstructor
@EqualsAndHashCode
@ToString
@Builder
@Setter
public class At {

    private int epoch;
    private int iteration;
    private String device;
    private int trainingThreadNum;
    private long javaThreadNum;

    public int epoch(){
        return epoch;
    }

    public int iteration(){
        return iteration;
    }

    public String device(){
        return device;
    }

    public int trainingThreadNum(){
        return trainingThreadNum;
    }

    public long javaThreadNum(){
        return javaThreadNum;
    }

}
