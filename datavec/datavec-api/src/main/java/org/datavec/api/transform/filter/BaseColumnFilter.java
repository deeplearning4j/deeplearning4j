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

package org.datavec.api.transform.filter;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.List;

/**Abstract class for filtering examples
 * based on the values in a
 * single column
 */
public abstract class BaseColumnFilter implements Filter {

    protected Schema schema;
    protected final String column;
    protected int columnIdx;

    protected BaseColumnFilter(String column) {
        this.column = column;
    }

    @Override
    public boolean removeExample(List<Writable> writables) {
        return removeExample(writables.get(columnIdx));
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        for (List<Writable> c : sequence) {
            if (removeExample(c))
                return true;
        }
        return false;
    }

    @Override
    public void setInputSchema(Schema schema) {
        this.schema = schema;
        this.columnIdx = schema.getIndexOfColumn(column);
    }

    /** Should the example or sequence be removed, based on the values from the specified column? */
    public abstract boolean removeExample(Writable writable);
}
