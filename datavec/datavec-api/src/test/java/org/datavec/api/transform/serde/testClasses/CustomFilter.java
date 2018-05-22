/*-
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

package org.datavec.api.transform.serde.testClasses;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Created by Alex on 11/01/2017.
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class CustomFilter implements Filter {

    private long someFilterArg;

    @Override
    public Schema transform(Schema inputSchema) {
        return null;
    }

    @Override
    public String outputColumnName() {
        return null;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[0];
    }

    @Override
    public String[] columnNames() {
        return new String[0];
    }

    @Override
    public String columnName() {
        return null;
    }

    @Override
    public boolean removeExample(Object example) {
        return false;
    }

    @Override
    public boolean removeSequence(Object sequence) {
        return false;
    }

    @Override
    public boolean removeExample(List<Writable> writables) {
        return false;
    }

    @Override
    public boolean removeSequence(List<List<Writable>> sequence) {
        return false;
    }

    @Override
    public void setInputSchema(Schema schema) {}

    @Override
    public Schema getInputSchema() {
        return null;
    }
}
