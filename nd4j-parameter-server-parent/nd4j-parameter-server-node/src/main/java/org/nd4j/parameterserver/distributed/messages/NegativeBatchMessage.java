package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * This is batch message, describing batch of NegativeSampling rounds
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
public class NegativeBatchMessage extends BaseVoidMessage {

    // indexes for current word
    protected int[] w1;

    // indixes for lastWord
    protected int[] w2;
}
