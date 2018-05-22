/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 *
 */

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
