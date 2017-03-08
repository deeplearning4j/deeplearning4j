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

package org.deeplearning4j.text.sentenceiterator;

import java.util.Collection;
import java.util.Iterator;

public class CollectionSentenceIterator extends BaseSentenceIterator {

    private Iterator<String> iter;
    private Collection<String> coll;

    public CollectionSentenceIterator(SentencePreProcessor preProcessor, Collection<String> coll) {
        super(preProcessor);
        this.coll = coll;
        iter = coll.iterator();
    }

    public CollectionSentenceIterator(Collection<String> coll) {
        this(null, coll);
    }

    @Override
    public String nextSentence() {
        String ret = iter.next();
        if (this.getPreProcessor() != null)
            ret = this.getPreProcessor().preProcess(ret);
        return ret;
    }

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }


    @Override
    public void reset() {
        iter = coll.iterator();
    }



}
