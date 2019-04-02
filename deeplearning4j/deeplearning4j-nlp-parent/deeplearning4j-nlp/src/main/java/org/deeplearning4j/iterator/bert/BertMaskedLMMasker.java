package org.deeplearning4j.iterator.bert;

import org.nd4j.base.Preconditions;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Alex on 03/04/2019.
 */
public class BertMaskedLMMasker implements BertSequenceMasker {
    public static final double DEFAULT_MASK_PROB = 0.15;
    public static final double DEFAULT_MASK_TOKEN_PROB = 0.8;
    public static final double DEFAULT_RANDOM_WORD_PROB = 0.1;

    protected final Random r;
    protected final double maskProb;
    protected final double maskTokenProb;
    protected final double randomWordProb;

    public BertMaskedLMMasker(){
        this(new Random(), DEFAULT_MASK_PROB, DEFAULT_MASK_TOKEN_PROB, DEFAULT_RANDOM_WORD_PROB);
    }

    public BertMaskedLMMasker(Random r, double maskProb, double maskTokenProb, double randomWordProb){
        Preconditions.checkArgument(maskProb > 0 && maskProb < 1, "Probability must be beteen 0 and 1, got %s", maskProb);
        this.r = r;
        this.maskProb = maskProb;
        this.maskTokenProb = maskTokenProb;
        this.randomWordProb = randomWordProb;
    }

    @Override
    public Pair<List<String>,boolean[]> maskSequence(List<String> input, String maskToken, List<String> vocabWords) {
        List<String> out = new ArrayList<>(input.size());
        boolean[] masked = new boolean[input.size()];
        for(int i=0; i<input.size(); i++ ){
            if(r.nextDouble() < maskProb){
                //Mask
                double d = r.nextDouble();
                if(d < maskTokenProb){
                    out.add(maskToken);
                } else if(d < maskTokenProb + randomWordProb){
                    //Randomly select a token...
                    String random = vocabWords.get(r.nextInt(vocabWords.size()));
                    out.add(random);
                } else {
                    //Keep existing token
                    out.add(input.get(i));
                }
                masked[i] = true;
            } else {
                //No change, keep existing
                out.add(input.get(i));
            }
        }
        return new Pair<>(out, masked);
    }
}
