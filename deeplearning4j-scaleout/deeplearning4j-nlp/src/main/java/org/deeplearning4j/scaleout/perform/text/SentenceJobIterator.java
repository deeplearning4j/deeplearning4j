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

package org.deeplearning4j.scaleout.perform.text;

import org.deeplearning4j.scaleout.job.Job;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

/**
 * Job iterator for sentences
 * @author Adam Gibson
 */
public class SentenceJobIterator implements JobIterator {
    private SentenceIterator iterator;

    public SentenceJobIterator(SentenceIterator iterator) {
        this.iterator = iterator;
    }

    @Override
    public Job next(String workerId) {
        return new Job(iterator.nextSentence(),workerId);
    }

    @Override
    public Job next() {
        return new Job(iterator.nextSentence(),"");
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public void reset() {
       iterator.reset();
    }
}
