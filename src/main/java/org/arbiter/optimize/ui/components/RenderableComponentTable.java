package org.arbiter.optimize.ui.components;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
public class RenderableComponentTable extends RenderableComponent {

    public static final String COMPONENT_TYPE = "simpletable";

    public RenderableComponentTable(){
        super(COMPONENT_TYPE);
        //No arg constructor for Jackson
    }

    public RenderableComponentTable(String[] header, String[][] table){
        super(COMPONENT_TYPE);
        this.header = header;
        this.table = table;
    }


    private String[] header;
    private String[][] table;




}
