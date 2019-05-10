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

package org.datavec.api.transform.serde.testClasses;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Created by Alex on 11/01/2017.
 */
@AllArgsConstructor
@NoArgsConstructor
@Data
public class CustomCondition implements Condition {

    private String someArg;

    @Override
    public Schema transform(Schema inputSchema) {
        return inputSchema;
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
    public boolean condition(List<Writable> list) {
        return false;
    }

    @Override
    public boolean condition(Object input) {
        return false;
    }

    @Override
    public boolean conditionSequence(List<List<Writable>> sequence) {
        return false;
    }

    @Override
    public boolean conditionSequence(Object sequence) {
        return false;
    }

    @Override
    public void setInputSchema(Schema schema) {}

    @Override
    public Schema getInputSchema() {
        return null;
    }
}
