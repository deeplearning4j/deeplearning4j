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

package org.nd4j.evaluation.meta;

import lombok.AllArgsConstructor;
import lombok.Data;

@AllArgsConstructor
@Data
public class Prediction {

    private int actualClass;
    private int predictedClass;
    private Object recordMetaData;

    @Override
    public String toString() {
        return "Prediction(actualClass=" + actualClass + ",predictedClass=" + predictedClass + ",RecordMetaData="
                        + recordMetaData + ")";
    }

    /**
     * Convenience method for getting the record meta data as a particular class (as an alternative to casting it manually).
     * NOTE: This uses an unchecked cast inernally.
     *
     * @param recordMetaDataClass Class of the record metadata
     * @param <T>                 Type to return
     */
    public <T> T getRecordMetaData(Class<T> recordMetaDataClass) {
        return (T) recordMetaData;
    }
}
