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

package org.datavec.api.transform.transform.floattransform;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.FloatMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.Writable;

/**
 *
 */
@EqualsAndHashCode(callSuper = true)
@Data
@NoArgsConstructor
public abstract class BaseFloatTransform extends BaseColumnTransform {

    public BaseFloatTransform(String column) {
        super(column);
    }

    public abstract Writable map(Writable writable);

    @Override
    public ColumnMetaData getNewColumnMetaData(String newColumnName, ColumnMetaData oldColumnMeta) {
        if (oldColumnMeta instanceof FloatMetaData) {
            ColumnMetaData meta = oldColumnMeta.clone();
            meta.setName(newColumnName);
            return meta;
        } else
            return new FloatMetaData(newColumnName);
    }



}
