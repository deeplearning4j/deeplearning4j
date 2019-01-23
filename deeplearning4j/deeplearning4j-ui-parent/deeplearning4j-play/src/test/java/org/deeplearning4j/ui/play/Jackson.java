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

package org.deeplearning4j.ui.play;

import org.junit.Test;
import org.nd4j.shade.jackson.databind.JavaType;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.Arrays;
import java.util.List;

public class Jackson {

    @Test
    public void test() throws Exception {

        ObjectMapper om = new ObjectMapper();
        List<String> l = Arrays.asList("a", "b", "c");
        String s = om.writeValueAsString(l);
        JavaType type = om.getTypeFactory().constructCollectionType(List.class, String.class);
        List<String> out = om.readValue(s, type);
        System.out.println(out);

    }

}
