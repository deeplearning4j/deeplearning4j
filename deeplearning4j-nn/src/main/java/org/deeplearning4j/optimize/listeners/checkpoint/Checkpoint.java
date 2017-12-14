package org.deeplearning4j.optimize.listeners.checkpoint;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor
@Data
public class Checkpoint {

    private int checkpointNum;
    private long timestamp;
    private int iteration;
    private int epoch;
    private String modelType;
    private String filename;

    public static Checkpoint fromFileString(String str){

    }

    public String toFileString(){

    }
}
