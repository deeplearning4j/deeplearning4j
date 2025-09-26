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

package org.deeplearning4j.text.documentiterator;

import java.io.InputStream;
import java.io.Serializable;


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
