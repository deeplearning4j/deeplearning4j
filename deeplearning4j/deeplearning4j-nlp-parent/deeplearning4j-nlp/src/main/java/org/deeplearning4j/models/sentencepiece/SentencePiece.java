/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.models.sentencepiece;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sentencepiece.enums.Algorithm;
import org.deeplearning4j.models.sentencepiece.impl.BinaryPairEncodingTrainer;
import org.deeplearning4j.models.sentencepiece.interfaces.Trainer;

import java.util.Iterator;

@Slf4j
@AllArgsConstructor
@NoArgsConstructor
public class SentencePiece {

    /**
     * Algorithm for this model
     */
    @Getter
    @Builder.Default
    private Algorithm algorithm = Algorithm.BPE;

    /**
     * Target vocabulary size
     */
    @Getter
    @Builder.Default
    private int vocabularySize = 32768;

    @Getter
    private SubwordVocabulary vocabulary;

    /**
     * Iterator with data
     */
    private transient Iterator<String> iterator;

    /**
     * This method trains model
     */
    public SubwordVocabulary fit() {
        Trainer trainer = null;
        switch (algorithm) {
            case BPE:
            default:
                trainer = new BinaryPairEncodingTrainer();
        }

        vocabulary = trainer.buildVocabulary(iterator);
        return vocabulary;
    }

    /**
     * This method converts String into sequence of IDs learned during training
     *
     * @param string
     * @return
     */
    public int[] vectorize(@NonNull String string) {
        return null;
    }

    /**
     * This vectors reconstructs String from given piece IDs
     * @param ids
     * @return
     */
    public String humanize(int... ids) {
        return null;
    }
}
