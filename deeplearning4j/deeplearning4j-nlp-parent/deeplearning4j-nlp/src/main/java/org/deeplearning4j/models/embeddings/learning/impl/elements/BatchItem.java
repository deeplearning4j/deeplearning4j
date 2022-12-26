/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.embeddings.learning.impl.elements;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.common.primitives.CounterMap;
import org.nd4j.shade.guava.collect.HashBasedTable;
import org.nd4j.shade.guava.collect.Table;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

public class BatchItem<T extends SequenceElement>  {
    private T word;
    private T lastWord;



    private int[] windowWords; // CBOW only
    private boolean[] wordStatuses;

    private long randomValue;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BatchItem<?> batchItem = (BatchItem<?>) o;
        return randomValue == batchItem.randomValue && Double.compare(batchItem.alpha, alpha) == 0 && windowWordsLength == batchItem.windowWordsLength && numLabel == batchItem.numLabel && Objects.equals(word, batchItem.word) && Objects.equals(lastWord, batchItem.lastWord) && Arrays.equals(windowWords, batchItem.windowWords) && Arrays.equals(wordStatuses, batchItem.wordStatuses);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(word, lastWord, randomValue, alpha, windowWordsLength, numLabel);
        result = 31 * result + Arrays.hashCode(windowWords);
        result = 31 * result + Arrays.hashCode(wordStatuses);
        return result;
    }

    @Override
    public String toString() {
        return "BatchItem{" +
                "word=" + word +
                ", lastWord=" + lastWord +
                ", windowWords=" + Arrays.toString(windowWords) +
                ", wordStatuses=" + Arrays.toString(wordStatuses) +
                ", randomValue=" + randomValue +
                ", alpha=" + alpha +
                ", windowWordsLength=" + windowWordsLength +
                ", numLabel=" + numLabel +
                '}';
    }

    private double alpha;
    private int windowWordsLength;

    private int numLabel;




    public BatchItem(T word, T lastWord, long randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
    }

    public BatchItem(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha, int numLabel) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
        this.numLabel = numLabel;
        this.windowWords = windowWords.clone();
        this.wordStatuses = wordStatuses.clone();

    }

    public BatchItem(T word, int[] windowWords, boolean[] wordStatuses, long randomValue, double alpha) {
        this.word = word;
        this.lastWord = lastWord;
        this.randomValue = randomValue;
        this.alpha = alpha;
        this.windowWords = windowWords.clone();
        this.wordStatuses = wordStatuses.clone();
    }

    public T getWord() {
        return word;
    }

    public void setWord(T word) {
        this.word = word;
    }

    public T getLastWord() {
        return lastWord;
    }

    public void setLastWord(T lastWord) {
        this.lastWord = lastWord;
    }

    public long getRandomValue() {
        return randomValue;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public int[] getWindowWords() {
        return windowWords;
    }

    public boolean[] getWordStatuses() {
        return wordStatuses;
    }

    public int getNumLabel() {
        return numLabel;
    }
}
