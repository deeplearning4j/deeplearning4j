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
package org.deeplearning4j.ui.components.component.style;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.ui.api.Style;

/** Style for Div components.
 *
 * @author Alex Black
 */
@NoArgsConstructor @Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class StyleDiv extends Style {

    /** Enumeration: possible values for float style option */
    public enum FloatValue {non, left, right, initial, inherit};

    private FloatValue floatValue;

    private StyleDiv(Builder builder){
        super(builder);
        this.floatValue = builder.floatValue;
    }


    public static class Builder extends Style.Builder<Builder>{

        private FloatValue floatValue;

        /** CSS float styling option */
        public Builder floatValue(FloatValue floatValue){
            this.floatValue = floatValue;
            return this;
        }

        public StyleDiv build(){
            return new StyleDiv(this);
        }
    }

}
