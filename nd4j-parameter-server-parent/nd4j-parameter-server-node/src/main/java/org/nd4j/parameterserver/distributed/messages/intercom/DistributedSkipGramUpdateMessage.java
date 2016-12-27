package org.nd4j.parameterserver.distributed.messages.intercom;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

/**
 * @author raver119@gmail.com
 */
public class DistributedSkipGramUpdateMessage extends BaseVoidMessage{

    protected int[] rows_syn0;
    protected int[] rows_syn1;
    protected float[] gradients;

    protected DistributedSkipGramUpdateMessage(){
        super(23);
    }

    public DistributedSkipGramUpdateMessage(@NonNull int rows_syn0[],@NonNull int rows_syn1[], @NonNull float gradients[]) {
        this();
        this.rows_syn0 = rows_syn0;
        this.rows_syn1 = rows_syn1;
        this.gradients = gradients;
    }

    @Override
    public void processMessage() {
        for(int r = 0; r < rows_syn0.length; r++) {
            // do syn0/syn1 updates
        }
    }
}
