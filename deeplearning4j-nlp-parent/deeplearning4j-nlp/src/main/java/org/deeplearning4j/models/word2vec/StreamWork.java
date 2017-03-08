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

import java.io.InputStream;
import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

public class StreamWork implements Serializable {
    private InputStreamCreator is;
    private AtomicInteger count = new AtomicInteger(0);


    public StreamWork(InputStreamCreator is, AtomicInteger count) {
        super();
        this.is = is;
        this.count = count;
    }

    public InputStream getIs() {
        return is.create();
    }

    public AtomicInteger getCount() {
        return count;
    }

    public void setCount(AtomicInteger count) {
        this.count = count;
    }

    public void countDown() {
        count.decrementAndGet();

    }



}
