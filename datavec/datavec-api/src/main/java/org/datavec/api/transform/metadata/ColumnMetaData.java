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

package org.datavec.api.transform.metadata;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.serde.legacy.LegacyMappingHelper;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * ColumnMetaData: metadata for each column. Used to define:
 * (a) the type of each column, and
 * (b) any restrictions on the allowable values in each column
 *
 * @author Alex Black
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class",
        defaultImpl = LegacyMappingHelper.ColumnMetaDataHelper.class)
public interface ColumnMetaData extends Serializable, Cloneable {

    /**
     * Get the name of the specified column
     * @return Name of the column
     */
    String getName();

    /**
     * Setter for the name
     * @param name
     */
    void setName(String name);

    /**
     * Get the type of column
     */
    ColumnType getColumnType();

    /**
     * Is the given Writable valid for this column, given the column type and any restrictions given by the
     * ColumnMetaData object?
     *
     * @param writable Writable to check
     * @return true if value, false if invalid
     */
    boolean isValid(Writable writable);

    /**
     * Is the given object valid for this column,
     * given the column type and any
     * restrictions given by the
     * ColumnMetaData object?
     *
     * @param input object to check
     * @return true if value, false if invalid
     */
    boolean isValid(Object input);

    /**
     *
     * @return
     */
    ColumnMetaData clone();
}
