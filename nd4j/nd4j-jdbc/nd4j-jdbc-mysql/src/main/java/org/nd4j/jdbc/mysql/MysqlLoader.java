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

package org.nd4j.jdbc.mysql;

import org.nd4j.jdbc.loader.impl.BaseLoader;

import javax.sql.DataSource;

/**
 * Mysql loader for ndarrays
 *
 * @author Adam Gibson
 */
public class MysqlLoader extends BaseLoader {

    public MysqlLoader(DataSource dataSource, String jdbcUrl, String tableName, String columnName) throws Exception {
        super(dataSource, jdbcUrl, tableName, columnName);
    }

    /**
     * Create an insert statement
     *
     * @return a new insert statement
     */
    @Override
    public String insertStatement() {
        return "INSERT INTO " + tableName + " VALUES(?,?)";
    }

    /**
     * Create an insert statement. This should be a templated query.
     * IE: Question mark at the end, we will take care of setting the proper value.
     *
     * @return a new insert statement
     */
    @Override
    public String loadStatement() {
        return "SELECT * FROM " + tableName + " WHERE " + this.idColumnName + " =?";


    }

    /**
     * Create an delete statement
     *
     * @return a new delete statement
     */
    @Override
    public String deleteStatement() {
        return "DELETE  FROM " + tableName + " WHERE " + this.idColumnName + " =?";

    }
}
