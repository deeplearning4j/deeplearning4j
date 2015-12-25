package org.arbiter.optimize.ui.components;

public class RenderElements {

    //Required for jackson
    public RenderElements(){ }

    private RenderableComponent[] renderableComponents;


    public RenderElements(RenderableComponent... renderableComponents) {
        this.renderableComponents = renderableComponents;
    }

    public RenderableComponent[] getRenderableComponents() {
        return this.renderableComponents;
    }

    public void setRenderableComponents(RenderableComponent[] renderableComponents) {
        this.renderableComponents = renderableComponents;
    }

    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof RenderElements)) return false;
        final RenderElements other = (RenderElements) o;
        if (!other.canEqual((Object) this)) return false;
        if (!java.util.Arrays.deepEquals(this.renderableComponents, other.renderableComponents)) return false;
        return true;
    }

    public int hashCode() {
        final int PRIME = 59;
        int result = 1;
        result = result * PRIME + java.util.Arrays.deepHashCode(this.renderableComponents);
        return result;
    }

    protected boolean canEqual(Object other) {
        return other instanceof RenderElements;
    }

    public String toString() {
        return "org.arbiter.optimize.ui.components.RenderElements(renderableComponents=" + java.util.Arrays.deepToString(this.renderableComponents) + ")";
    }
}
