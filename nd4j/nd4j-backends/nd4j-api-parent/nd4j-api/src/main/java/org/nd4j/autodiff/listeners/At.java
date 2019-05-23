package org.nd4j.autodiff.listeners;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.ToString;

@AllArgsConstructor
@EqualsAndHashCode
@ToString
public class At {

    private int epoch;
    private int iteration;
    private String device;
    private int trainingThreadNum;
    private int javaThreadNum;

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

    public int javaThreadNum(){
        return javaThreadNum;
    }

}
