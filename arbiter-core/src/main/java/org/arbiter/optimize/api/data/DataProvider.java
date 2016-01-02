/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.arbiter.optimize.api.data;

import java.util.Map;

public interface DataProvider<D> {

    /** Get training data given some parameters for the data. Data parameters map is used to specify things like batch
     * size data preprocessing
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    D trainData(Map<String, Object> dataParameters);

    /** Get training data given some parameters for the data. Data parameters map is used to specify things like batch
     * size data preprocessing
     * @param dataParameters Parameters for data. May be null or empty for default data
     * @return training data
     */
    D testData(Map<String, Object> dataParameters);

}
