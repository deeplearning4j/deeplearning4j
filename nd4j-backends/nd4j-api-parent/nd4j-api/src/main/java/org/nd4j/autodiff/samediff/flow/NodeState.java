package org.nd4j.autodiff.samediff.flow;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.primitives.Pair;

/**
 * This class describe Node state during execution time.
 *
 * @author raver119@gmail.com
 */
@Data
public class NodeState {
    private String nodeName;
    private boolean active = true;
    private int activeBranch = 0;
    private boolean executed = false;
    private long numCycles = 0;

    private int rewindPosition = -1;
    private String rewindNode;

    public NodeState(@NonNull String nodeName) {
        this.nodeName = nodeName;
    }

    public void incrementNumberOfCycles() {
        numCycles++;
    }

    public long getNumberOfCycles() {
        return numCycles;
    }
}
