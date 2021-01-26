/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.iterator.bert;

import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A standard/default {@link BertSequenceMasker}. Implements masking as per the BERT paper:
 * <a href="https://arxiv.org/abs/1810.04805">https://arxiv.org/abs/1810.04805</a>
 * That is, each token is chosen to be masked independently with some probability "maskProb".
 * For tokens that are masked, 3 possibilities:<br>
 * 1. They are replaced with the mask token (such as "[MASK]") in the input, with probability "maskTokenProb"<br>
 * 2. They are replaced with a random word from the vocabulary, with probability "randomTokenProb"<br>
 * 3. They are are left unmodified with probability 1.0 - maskTokenProb - randomTokenProb<br>
 *
 * @author Alex Black
 */
public class BertMaskedLMMasker implements BertSequenceMasker {
    public static final double DEFAULT_MASK_PROB = 0.15;
    public static final double DEFAULT_MASK_TOKEN_PROB = 0.8;
    public static final double DEFAULT_RANDOM_WORD_PROB = 0.1;

    protected final Random r;
    protected final double maskProb;
    protected final double maskTokenProb;
    protected final double randomTokenProb;

    /**
     * Create a BertMaskedLMMasker with all default probabilities
     */
    public BertMaskedLMMasker(){
        this(new Random(), DEFAULT_MASK_PROB, DEFAULT_MASK_TOKEN_PROB, DEFAULT_RANDOM_WORD_PROB);
    }

    /**
     * See: {@link BertMaskedLMMasker} for details.
     * @param r                 Random number generator
     * @param maskProb          Probability of masking each token
     * @param maskTokenProb     Probability of replacing a selected token with the mask token
     * @param randomTokenProb    Probability of replacing a selected token with a random token
     */
    public BertMaskedLMMasker(Random r, double maskProb, double maskTokenProb, double randomTokenProb){
        Preconditions.checkArgument(maskProb > 0 && maskProb < 1, "Probability must be beteen 0 and 1, got %s", maskProb);
        Preconditions.checkState(maskTokenProb >=0 && maskTokenProb <= 1.0, "Mask token probability must be between 0 and 1, got %s", maskTokenProb);
        Preconditions.checkState(randomTokenProb >=0 && randomTokenProb <= 1.0, "Random token probability must be between 0 and 1, got %s", randomTokenProb);
        Preconditions.checkState(maskTokenProb + randomTokenProb <= 1.0, "Sum of maskTokenProb (%s) and randomTokenProb (%s) must be <= 1.0, got sum is %s",
                maskTokenProb, randomTokenProb, (maskTokenProb + randomTokenProb));
        this.r = r;
        this.maskProb = maskProb;
        this.maskTokenProb = maskTokenProb;
        this.randomTokenProb = randomTokenProb;
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
                } else if(d < maskTokenProb + randomTokenProb){
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
