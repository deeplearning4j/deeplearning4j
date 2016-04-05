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

import io.skymind.ui.api.Component;
import io.skymind.ui.components.table.style.TableStyle;
import lombok.Data;
import lombok.EqualsAndHashCode;

@EqualsAndHashCode(callSuper = true)
@Data
public class ComponentTable extends Component {
    public static final String COMPONENT_TYPE = "component_table";

    private String title;
    private String[] header;
    private String[][] table;

    public ComponentTable(){
        super(COMPONENT_TYPE, null);
        //No arg constructor for Jackson
    }

    public ComponentTable(Builder builder){
        super(COMPONENT_TYPE, builder.style);
        this.header = builder.header;
        this.table = builder.table;
    }

    public ComponentTable(String[] header, String[][] table, TableStyle style){
        super(COMPONENT_TYPE, style);
        this.header = header;
        this.table = table;
    }

    public static class Builder {

        private TableStyle style;
        private String[] header;
        private String[][] table;

        public Builder(TableStyle style){
            this.style = style;
        }

        public Builder header(String... header){
            this.header = header;
            return this;
        }

        public Builder table(String[][] table){
            this.table = table;
            return this;
        }

        public ComponentTable build(){
            return new ComponentTable(this);
        }

    }



}
