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

package org.datavec.api.transform.transform.string;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.StringMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;


/**
 * Abstract String column transform
 */
@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
public abstract class BaseStringTransform extends BaseColumnTransform {

    public BaseStringTransform(String column) {
        super(column);
    }

    /**
     * Transform the writable in to a
     * string
     * @param writable the writable to transform
     * @return the string form of this writable
     */
    public abstract Text map(Writable writable);

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new StringMetaData(newName);
    }
}
