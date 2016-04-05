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
package org.deeplearning4j.ui.components.text;

import io.skymind.ui.api.Component;
import io.skymind.ui.components.text.style.TextStyle;
import lombok.Data;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
@Data
public class ComponentText extends Component {
    public static final String COMPONENT_TYPE = "component_text";
    private String string;

    public ComponentText(){
        super(COMPONENT_TYPE, null);
        //No arg constructor for Jackson deserialization
        string = null;
    }

    public ComponentText(String string, TextStyle style){
        super(COMPONENT_TYPE, style);
        this.string = string;
    }


    @Override
    public String toString(){
        return "ComponentText(" + string + ")";
    }

    public static class Builder {

        private TextStyle style;
        private String text;

        public Builder(String text, TextStyle style){
            this.text = text;
            this.style = style;
        }


    }

}
