/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.api.transform.metadata;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.writable.Writable;

import java.io.Serializable;

/**
 * ColumnMetaData: metadata for each column. Used to define:
 * (a) the type of each column, and
 * (b) any restrictions on the allowable values in each column
 *
 * @author Alex Black
 */
public interface ColumnMetaData extends Serializable, Cloneable {

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

    ColumnMetaData clone();
}
