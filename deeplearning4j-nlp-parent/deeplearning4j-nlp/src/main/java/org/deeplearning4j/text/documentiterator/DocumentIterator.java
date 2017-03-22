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

package org.deeplearning4j.text.documentiterator;

import java.io.InputStream;
import java.io.Serializable;


/**
 * Document Iterator: iterate over input streams
 * @author Adam Gibson
 *
 */
public interface DocumentIterator extends Serializable {



    /**
     * Get the next document
     * @return the input stream for the next document
     */
    InputStream nextDocument();

    /**
     * Whether there are anymore documents in the iterator
     * @return whether there are anymore documents
     * in the iterator
     */
    boolean hasNext();

    /**
     * Reset the iterator to the beginning
     */
    void reset();



}
