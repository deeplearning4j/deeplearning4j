/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.ui.nearestneighbors.word2vec;

import java.io.Serializable;

/**
 * @author Adam Gibson
 */
public class NearestNeighborsQuery implements Serializable {
    private String word;
    private int numWords;

    public NearestNeighborsQuery(String word, int numWords) {
        this.word = word;
        this.numWords = numWords;
    }

    public NearestNeighborsQuery() {
    }

    public String getWord() {
        return word;
    }

    public void setWord(String word) {
        this.word = word;
    }

    public int getNumWords() {
        return numWords;
    }

    public void setNumWords(int numWords) {
        this.numWords = numWords;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        NearestNeighborsQuery that = (NearestNeighborsQuery) o;

        if (numWords != that.numWords) return false;
        return !(word != null ? !word.equals(that.word) : that.word != null);

    }

    @Override
    public int hashCode() {
        int result = word != null ? word.hashCode() : 0;
        result = 31 * result + numWords;
        return result;
    }
}
