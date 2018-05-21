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
