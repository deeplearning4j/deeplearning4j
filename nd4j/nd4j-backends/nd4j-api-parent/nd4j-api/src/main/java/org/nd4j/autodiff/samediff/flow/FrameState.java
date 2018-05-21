package org.nd4j.autodiff.samediff.flow;

import lombok.Data;
import lombok.NonNull;

/**
 * This class is a holder for state of loops imported from TensorFlow, via frame_name
 *
 * @author raver119@gmail.com
 */
@Data
public class FrameState {
    private String name;
    private long iterations = 0;
    private boolean rewindPlanned = false;
    private int rewindPosition = -1;

    private int numberOfEntries = 0;
    private int numberOfExits = 0;
    private boolean active = false;
    private int numberOfCycles;


    public FrameState(@NonNull String frame_name) {
        this.name = frame_name;
    }

    /**
     * This method returns number of cycles for this frame
     * @return
     */
    public int getNumberOfCycles() {
        return numberOfCycles;
    }

    /**
     * This method increments number of cycles by 1
     */
    public void incrementNumberOfCycles() {
        numberOfCycles++;
    }
}
