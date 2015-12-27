package org.arbiter.optimize.ui.components;

import lombok.AllArgsConstructor;
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
