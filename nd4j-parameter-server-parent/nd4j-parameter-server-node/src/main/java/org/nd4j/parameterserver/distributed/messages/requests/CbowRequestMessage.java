package org.nd4j.parameterserver.distributed.messages.requests;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.RequestMessage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.VoidMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

/**
 * @author raver119@gmail.com
 */
@Data
@Slf4j
public class CbowRequestMessage extends BaseVoidMessage implements TrainingMessage, RequestMessage {
    protected byte counter = 1;

    long frameId;

    protected int w1;
    protected int[] syn0rows;
    protected int[] syn1rows;
    protected double alpha;
    protected long nextRandom;
    protected int negSamples;
    protected byte[] codes;

    protected int[] negatives;

    public CbowRequestMessage(@NonNull int[] syn0rows, @NonNull int[] syn1rows, int w1, byte[] codes, int negSamples,
                    double alpha, long nextRandom) {
        this.syn0rows = syn0rows;
        this.syn1rows = syn1rows;
        this.w1 = w1;
        this.alpha = alpha;
        this.nextRandom = nextRandom;
        this.negSamples = negSamples;
        this.codes = codes;


        this.setTaskId(BasicSequenceProvider.getInstance().getNextValue());
    }

    @Override
    @SuppressWarnings("unchecked")
    public void processMessage() {
        // we just pick training here
        TrainingDriver<CbowRequestMessage> cbt = (TrainingDriver<CbowRequestMessage>) trainer;
        cbt.startTraining(this);
    }

    @Override
    public boolean isJoinSupported() {
        return true;
    }

    @Override
    public void joinMessage(VoidMessage message) {
        // TODO: apply proper join handling here
        counter++;
    }
}
