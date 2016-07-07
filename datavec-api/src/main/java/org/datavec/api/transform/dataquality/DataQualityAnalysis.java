package org.datavec.api.transform.dataquality;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.schema.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.datavec.api.transform.dataquality.columns.ColumnQuality;

import java.util.List;

/**A report outlining number of invalid and missing features
 */
@AllArgsConstructor @Data
public class DataQualityAnalysis {

    private Schema schema;
    private List<ColumnQuality> columnQualityList;


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
                .append(String.format("%-15s","type"))
                .append(String.format("%-10s","quality"))
                .append("details").append("\n");

        for( int i=0; i<nCol; i++ ){
            String colName = schema.getName(i);
            ColumnType type = schema.getType(i);
            ColumnQuality columnQuality = columnQualityList.get(i);
            boolean pass = columnQuality.getCountInvalid() == 0L && columnQuality.getCountMissing() == 0L;
            String paddedName = String.format("%-"+(maxNameLength+8)+"s","\"" + colName + "\"");
            sb.append(String.format("%-6d",i))
                    .append(paddedName)
                    .append(String.format("%-15s",type))
                    .append(String.format("%-10s",(pass ? "ok" : "FAIL")))
                    .append(columnQuality).append("\n");
        }

        return sb.toString();
    }

}
