package io.skymind.echidna.api.analysis;

import io.skymind.echidna.api.ColumnType;
import io.skymind.echidna.api.analysis.columns.ColumnAnalysis;
import lombok.AllArgsConstructor;
import lombok.Data;
import io.skymind.echidna.api.schema.Schema;

import java.io.Serializable;
import java.util.List;

/**
 * The DataAnalysis class represents analysis (summary statistics) for a data set.
 *
 * @author Alex Black
 */
@AllArgsConstructor @Data
public class DataAnalysis implements Serializable {

    private Schema schema;
    private List<ColumnAnalysis> columnAnalysis;

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        int nCol = schema.numColumns();

        int maxNameLength = 0;
        for(String s :schema.getColumnNames()){
            maxNameLength = Math.max(maxNameLength,s.length());
        }

        //Header:
        sb.append(String.format("%-6s","idx")).append(String.format("%-"+(maxNameLength+8)+"s","name"))
                .append(String.format("%-15s","type")).append("analysis").append("\n");

        for( int i=0; i<nCol; i++ ){
            String colName = schema.getName(i);
            ColumnType type = schema.getType(i);
            ColumnAnalysis analysis = columnAnalysis.get(i);
            String paddedName = String.format("%-"+(maxNameLength+8)+"s","\"" + colName + "\"");
            sb.append(String.format("%-6d",i))
                    .append(paddedName)
                    .append(String.format("%-15s",type))
                    .append(analysis).append("\n");
        }

        return sb.toString();
    }

    public ColumnAnalysis getColumnAnalysis(String column){
        return columnAnalysis.get(schema.getIndexOfColumn(column));
    }
}
