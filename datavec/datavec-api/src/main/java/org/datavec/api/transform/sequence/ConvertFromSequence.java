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

package org.datavec.api.transform.sequence;


import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.ArrayList;
import java.util.List;

/**
 * Split up the values in sequences to a set of individual values.<br>
 * i.e., sequences are split up, such that each time step in the sequence is treated as a separate example
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"inputSchema"})
@JsonIgnoreProperties({"inputSchema"})
public class ConvertFromSequence {

    private SequenceSchema inputSchema;

    public ConvertFromSequence() {

    }

    public Schema transform(SequenceSchema schema) {
        List<ColumnMetaData> meta = new ArrayList<>(schema.getColumnMetaData());

        return new Schema(meta);
    }

    @Override
    public String toString() {
        return "ConvertFromSequence()";
    }

}
