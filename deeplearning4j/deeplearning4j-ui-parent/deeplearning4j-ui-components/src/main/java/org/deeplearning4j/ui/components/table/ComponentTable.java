/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.components.table;


import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.ui.api.Component;
import org.deeplearning4j.ui.components.table.style.StyleTable;
import org.nd4j.shade.jackson.annotation.JsonInclude;

/**
 * Simple 2d table for text,
 *
 * @author Alex Black
 */
@EqualsAndHashCode(callSuper = true)
@Data
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ComponentTable extends Component {
    public static final String COMPONENT_TYPE = "ComponentTable";

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
