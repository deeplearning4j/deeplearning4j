/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.dataset.api.iterator.cache;

import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by anton on 7/16/16.
 */
public interface DataSetCache {
    /**
     * Check is given namespace has complete cache of the data set
     * @param namespace
     * @return true if namespace is fully cached
     */
    boolean isComplete(String namespace);

    /**
     * Sets the flag indicating whether given namespace is fully cached
     * @param namespace
     * @param value
     */
    void setComplete(String namespace, boolean value);

    DataSet get(String key);

    void put(String key, DataSet dataSet);

    boolean contains(String key);
}
