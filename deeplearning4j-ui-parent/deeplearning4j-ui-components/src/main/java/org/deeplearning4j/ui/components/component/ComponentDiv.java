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
package org.deeplearning4j.ui.components.component;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.api.Style;

/**
 * Div component (as in, HTML div)
 *
 * @author Alex Black
 */
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ComponentDiv extends Component {
    public static final String COMPONENT_TYPE = "ComponentDiv";

    private Component[] components;

    public ComponentDiv(){
        super(COMPONENT_TYPE,null);
    }


    public ComponentDiv(Style style, Component... components) {
        super(COMPONENT_TYPE, style);
        this.components = components;
    }
}
