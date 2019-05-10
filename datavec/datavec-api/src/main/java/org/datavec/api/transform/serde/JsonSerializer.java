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

package org.datavec.api.transform.serde;

import org.datavec.api.transform.Transform;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.reduce.IAssociativeReducer;
import org.datavec.api.transform.sequence.SequenceComparator;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.Arrays;

/**
 * Serializer used for converting objects (Transforms, Conditions, etc) to JSON format
 *
 * @author Alex Black
 */
public class JsonSerializer extends BaseSerializer {

    private ObjectMapper om;

    public JsonSerializer() {
        this.om = JsonMappers.getMapper();
    }

    @Override
    public ObjectMapper getObjectMapper() {
        return om;
    }
}
