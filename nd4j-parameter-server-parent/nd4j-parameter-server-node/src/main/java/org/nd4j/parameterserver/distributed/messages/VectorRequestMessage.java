package org.nd4j.parameterserver.distributed.messages;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * This message requests full weights vector for specified index
 *
 * Client -> Shard version
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class VectorRequestMessage extends BaseVoidMessage {

    protected int rowIndex;

    public VectorRequestMessage(int rowIndex) {
        super(7);
        this.rowIndex = rowIndex;
    }
}
