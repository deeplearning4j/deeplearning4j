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

package org.datavec.api.transform.sequence.comparator;

import lombok.Data;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * A comparator for comparing
 * String values in a single column
 *
 * @author Alex
 */
@Data
public class StringComparator extends BaseColumnComparator {

    /**
     * @param columnName Name of the column in which to compare values
     */
    public StringComparator(@JsonProperty("columnName") String columnName) {
        super(columnName);
    }

    @Override
    protected int compare(Writable w1, Writable w2) {
        return w1.toString().compareTo(w2.toString());
    }
}
