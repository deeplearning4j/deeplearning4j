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
package org.deeplearning4j.ui.components.table;


import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.components.table.style.StyleTable;

/**
 * Simple 2d table for text,
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ComponentTable extends Component {
    public static final String COMPONENT_TYPE = "component_table";

    private String title;
    private String[] header;
    private String[][] content;

    public ComponentTable() {
        super(COMPONENT_TYPE, null);
        //No arg constructor for Jackson
    }

    public ComponentTable(Builder builder) {
        super(COMPONENT_TYPE, builder.style);
        this.header = builder.header;
        this.content = builder.content;
    }

    public ComponentTable(String[] header, String[][] table, StyleTable style) {
        super(COMPONENT_TYPE, style);
        this.header = header;
        this.content = table;
    }

    public static class Builder {

        private StyleTable style;
        private String[] header;
        private String[][] content;

        public Builder(StyleTable style) {
            this.style = style;
        }

        /**
         * @param header Header values for the table
         */
        public Builder header(String... header) {
            this.header = header;
            return this;
        }

        /**
         * Content for the table, as 2d String[]
         */
        public Builder content(String[][] content) {
            this.content = content;
            return this;
        }

        public ComponentTable build() {
            return new ComponentTable(this);
        }

    }


}
