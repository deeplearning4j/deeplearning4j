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

package org.datavec.api.transform.transform;

import lombok.Data;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.ArrayList;
import java.util.List;

/**
 * BaseTransform: an
 * abstract transform class, that handles transforming
 * sequences by transforming
 * each example individually
 *
 * @author Alex Black
 */
@JsonIgnoreProperties({"inputSchema"})
@Data
public abstract class BaseTransform implements Transform {

    protected Schema inputSchema;

    @Override
    public void setInputSchema(Schema inputSchema) {
        this.inputSchema = inputSchema;
    }

    @Override
    public Schema getInputSchema() {
        return inputSchema;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {

        List<List<Writable>> out = new ArrayList<>(sequence.size());
        for (List<Writable> c : sequence) {
            out.add(map(c));
        }
        return out;
    }

    @Override
    public abstract String toString();
}
