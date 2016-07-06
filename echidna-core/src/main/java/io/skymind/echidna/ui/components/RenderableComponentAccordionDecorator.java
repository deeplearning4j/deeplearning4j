package io.skymind.echidna.ui.components;

import lombok.Data;
import lombok.EqualsAndHashCode;

/**Renderable component within an accordion-type component
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class RenderableComponentAccordionDecorator extends RenderableComponent {
    public static final String COMPONENT_TYPE = "accordion";

    private String title;
    private boolean defaultCollapsed;
    private RenderableComponent[] innerComponents;

    public RenderableComponentAccordionDecorator(){
        super(COMPONENT_TYPE);
        //No arg constructor for Jackson
    }

    public RenderableComponentAccordionDecorator(String title, boolean defaultCollapsed, RenderableComponent... innerComponents) {
        super(COMPONENT_TYPE);
        this.title = title;
        this.defaultCollapsed = defaultCollapsed;
        this.innerComponents = innerComponents;
    }
}
