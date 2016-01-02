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
