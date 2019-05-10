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

package org.datavec.api.util.jdbc;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Iterator;
import org.apache.commons.dbutils.ResultSetIterator;

/**
 * Encapsulation of ResultSetIterator to allow resetting
 *
 * @author Adrien Plagnol
 */
public class ResettableResultSetIterator implements Iterator<Object[]> {

    private ResultSet rs;
    private ResultSetIterator base;

    public ResettableResultSetIterator(ResultSet rs) {
        this.rs = rs;
        this.base = new ResultSetIterator(rs);
    }

    public void reset() {
        try {
            this.rs.beforeFirst();
        } catch (SQLException e) {
            throw new RuntimeException("Could not reset ResultSetIterator", e);
        }
    }

    @Override
    public boolean hasNext() {
        return this.base.hasNext();
    }

    @Override
    public Object[] next() {
        return base.next();
    }

    @Override
    public void remove() {
        base.remove();
    }
}
