package org.arbiter.optimize.ui.components;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.arbiter.optimize.ui.rendering.RenderElements;
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
