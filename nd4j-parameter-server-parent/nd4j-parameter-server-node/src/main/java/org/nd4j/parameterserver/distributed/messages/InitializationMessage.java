package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author raver119@gmail.com
 */
@NoArgsConstructor
@Builder
@Data
public class InitializationMessage extends BaseVoidMessage {

    protected int vectorLength;
    protected int numWords;
    protected long seed;
    protected boolean useHs;
    protected boolean useNeg;
    protected int columnsPerShard;

    public InitializationMessage(int vectorLength, int numWords, long seed, boolean useHs, boolean useNeg, int columnsPerShard) {
        super(4);
    }
}
