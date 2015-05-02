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

package org.deeplearning4j.datasets.creator;

import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Base interface for creating datasetiterators
 * @author Adam Gibson
 */
public interface DataSetIteratorFactory {
    String NAME_SPACE = "org.deeplearning4j.datasets.creator";
    String FACTORY_KEY = NAME_SPACE + ".datasetiteratorkey";
    /**
     * Create a dataset iterator
     * @return
     */
    DataSetIterator create();

}
