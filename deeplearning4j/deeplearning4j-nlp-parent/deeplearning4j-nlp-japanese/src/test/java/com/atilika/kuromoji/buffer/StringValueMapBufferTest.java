/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package com.atilika.kuromoji.buffer;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.util.TreeMap;

import static org.junit.Assert.assertEquals;

public class StringValueMapBufferTest extends BaseDL4JTest {

    @Test
    public void testInsertIntoMap() throws Exception {
        TreeMap<Integer, String> input = new TreeMap<>();

        input.put(1, "hello");
        input.put(2, "日本");
        input.put(0, "Bye");

        StringValueMapBuffer values = new StringValueMapBuffer(input);

        assertEquals("Bye", values.get(0));
        assertEquals("hello", values.get(1));
        assertEquals("日本", values.get(2));
    }
}
