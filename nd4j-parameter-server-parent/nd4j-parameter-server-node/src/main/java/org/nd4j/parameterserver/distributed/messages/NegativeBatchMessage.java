package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.SerializationUtils;

/**
 * This is batch message, describing batch of NegativeSampling rounds
 *
 * @author raver119@gmail.com
 */
@Data
public class NegativeBatchMessage extends BaseVoidMessage {

    public NegativeBatchMessage() {
        super(0);
    }

    // indexes for current word
    protected int[] w1;

    // indixes for lastWord
    protected int[] w2;
}
