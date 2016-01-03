/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
 *
 */
package org.arbiter.optimize.ui.components;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;

public class RenderableComponentsJSONTest {

    @Test
    public void testString() throws Exception {
        RenderableComponent component = new RenderableComponentString("This is a string");

        ObjectMapper mapper = new ObjectMapper();
        String str = mapper.writeValueAsString(component);
        System.out.println(str);
    }

    @Test
    public void testRenderElements() throws Exception {

        RenderElements elements = new RenderElements(
                new RenderableComponentString("This is a string")
        );

        ObjectMapper mapper = new ObjectMapper();
        String str = mapper.writeValueAsString(elements);
        System.out.println(str);

        RenderElements e2 = mapper.readValue(str,RenderElements.class);
        System.out.println(e2);
    }

}
