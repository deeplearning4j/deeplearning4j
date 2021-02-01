/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.models.embeddings.learning.impl.elements;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

public class BatchItem<T extends SequenceElement>  {
    private T word;
    private T lastWord;

    private int[] windowWords; // CBOW only
    private boolean[] wordStatuses;

    private long randomValue;
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
