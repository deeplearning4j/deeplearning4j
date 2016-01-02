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

import lombok.Data;

@Data
public class RenderableComponentTable extends RenderableComponent {

    public static final String COMPONENT_TYPE = "simpletable";

    private String title;
    private String[] header;
    private String[][] table;

    public RenderableComponentTable(){
        super(COMPONENT_TYPE);
        //No arg constructor for Jackson
    }

    public RenderableComponentTable(String[] header, String[][] table){
        this(null,header,table);
    }

    public RenderableComponentTable(String title, String[] header, String[][] table){
        super(COMPONENT_TYPE);
        this.title = title;
        this.header = header;
        this.table = table;
    }




}
