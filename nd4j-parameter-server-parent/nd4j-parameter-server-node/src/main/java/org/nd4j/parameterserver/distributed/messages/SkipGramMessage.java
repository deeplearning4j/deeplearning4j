package org.nd4j.parameterserver.distributed.messages;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.lang3.SerializationUtils;

/**
 * This is batch message, describing simple SkipGram round
 *
 * @author raver119@gmail.com
 */
@Data
public class SkipGramMessage extends BaseVoidMessage {

    // current word & lastWord
    protected int w1;
    protected int w2;

    // following fields are for hierarchic softmax
    // points & codes for current word
    protected int[] points;
    protected byte[] codes;

    public SkipGramMessage(int w1, int w2, int[] points, byte[] codes) {
        super(0);
        this.w1 = w1;
        this.w2 = w2;
        this.points = points;
        this.codes = codes;
    }
}
