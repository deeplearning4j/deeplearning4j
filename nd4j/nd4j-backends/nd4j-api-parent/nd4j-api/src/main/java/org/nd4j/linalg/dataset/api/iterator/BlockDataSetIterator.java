/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset.api.iterator;

import org.nd4j.linalg.dataset.api.DataSet;

/**
 * This abstraction provides block access to underlying DataSetIterator
 * @author raver119@gmail.com
 */
public interface BlockDataSetIterator {

    /**
     * This method checks, if underlying iterator has at least 1 DataSet
     * @return
     */
    boolean hasAnything();

    /**
     * This method tries to fetch specified number of DataSets and returns them
     * @param maxDatasets
     * @return
     */
    DataSet[] next(int maxDatasets);
}
