package org.deeplearning4j.optimize.listeners.checkpoint;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;

/**
 * A model checkpoint, used with {@link CheckpointListener}
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class Checkpoint implements Serializable {

    private int checkpointNum;
    private long timestamp;
    private int iteration;
    private int epoch;
    private String modelType;
    private String filename;

    public static String getFileHeader(){
        return "checkpointNum,timestamp,iteration,epoch,modelType,filename";
    }

    public static Checkpoint fromFileString(String str){
        String[] split = str.split(",");
        if(split.length != 6){
            throw new IllegalStateException("Cannot parse checkpoint entry: expected 6 entries, got " + split.length
                    + " - values = " + Arrays.toString(split));
        }
        return new Checkpoint(
                Integer.parseInt(split[0]),
                Long.parseLong(split[1]),
                Integer.parseInt(split[2]),
                Integer.parseInt(split[3]),
                split[4],
                split[5]);
    }

    public String toFileString(){
        return checkpointNum + "," + timestamp + "," + iteration + "," + epoch + "," + modelType + "," + filename;
    }
}
