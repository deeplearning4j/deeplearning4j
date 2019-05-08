/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.datavec.python;

import org.datavec.api.transform.Transform;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.JsonSerializer;
import org.datavec.api.transform.serde.YamlSerializer;
import static org.junit.Assert.assertEquals;
import org.junit.Test;

public class TestSerde {

    public static YamlSerializer y = new YamlSerializer();
    public static JsonSerializer j = new JsonSerializer();

    @Test
    public void testBasicSerde() throws Exception{
        Schema schema = new Schema.Builder()
                .addColumnInteger("col1")
                .addColumnFloat("col2")
                .addColumnString("col3")
                .addColumnDouble("col4")
                .build();

        Transform t = new PythonTransform(
                "col1+=3\ncol2+=2\ncol3+='a'\ncol4+=2.0",
                schema
        );

        String yaml = y.serialize(t);
        String json = j.serialize(t);

        Transform t2 = y.deserializeTransform(json);
        Transform t3 = j.deserializeTransform(json);
        assertEquals(t, t2);
        assertEquals(t, t3);
    }

}
