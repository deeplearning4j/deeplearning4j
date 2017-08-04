/*-
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

package org.deeplearning4j.models.word2vec;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Vocab work meant for use with the vocab actor
 *
 *
 * @author Adam Gibson
 */
public class VocabWork implements Serializable {

    private AtomicInteger count = new AtomicInteger(0);
    private String work;
    private boolean stem = false;
    private List<String> label;


    public VocabWork(AtomicInteger count, String work, boolean stem) {
        this(count, work, stem, "");
    }



    public VocabWork(AtomicInteger count, String work, boolean stem, String label) {
        this(count, work, stem, Arrays.asList(label));
    }

    public VocabWork(AtomicInteger count, String work, boolean stem, List<String> label) {
        this.count = count;
        this.work = work;
        this.stem = stem;
        this.label = label;
    }

    public AtomicInteger getCount() {
        return count;
    }

    public void setCount(AtomicInteger count) {
        this.count = count;
    }

    public String getWork() {
        return work;
    }

    public void setWork(String work) {
        this.work = work;
    }

    public void increment() {
        count.incrementAndGet();
    }

    public boolean isStem() {
        return stem;
    }

    public void setStem(boolean stem) {
        this.stem = stem;
    }

    public List<String> getLabel() {
        return label;
    }

    public void setLabel(List<String> label) {
        this.label = label;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof VocabWork))
            return false;

        VocabWork vocabWork = (VocabWork) o;

        if (stem != vocabWork.stem)
            return false;
        if (count != null ? !count.equals(vocabWork.count) : vocabWork.count != null)
            return false;
        if (label != null ? !label.equals(vocabWork.label) : vocabWork.label != null)
            return false;
        return !(work != null ? !work.equals(vocabWork.work) : vocabWork.work != null);

    }

    @Override
    public int hashCode() {
        int result = count != null ? count.hashCode() : 0;
        result = 31 * result + (work != null ? work.hashCode() : 0);
        result = 31 * result + (stem ? 1 : 0);
        result = 31 * result + (label != null ? label.hashCode() : 0);
        return result;
    }
}
