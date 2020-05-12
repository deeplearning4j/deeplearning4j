/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.imports.TFGraphs;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.Map;

@AllArgsConstructor
@Data
public class TestCase {
    public String modelName;
    public Map<String,String> inputs;  //Key: variable name, values: filename (.csv)
    public Map<String,String> outputs;
    public Map<String, DataType> datatypes;
}
