package org.nd4j.etl4j.api.transform.schema;

import org.nd4j.etl4j.api.transform.ColumnType;
import org.nd4j.etl4j.api.transform.metadata.ColumnMetaData;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.util.List;

/**
 * Created by Alex on 11/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class SequenceSchema extends Schema {
    private final Integer minSequenceLength;
    private final Integer maxSequenceLength;

    public SequenceSchema(List<String> columnNames, List<ColumnMetaData> columnMetaData){
        this(columnNames,columnMetaData,null,null);
    }

    public SequenceSchema(List<String> columnNames, List<ColumnMetaData> columnMetaData,
                          Integer minSequenceLength, Integer maxSequenceLength) {
        super(columnNames, columnMetaData);
        this.minSequenceLength = minSequenceLength;
        this.maxSequenceLength = maxSequenceLength;
    }

    private SequenceSchema(Builder builder){
        super(builder);
        this.minSequenceLength = builder.minSequenceLength;
        this.maxSequenceLength = builder.maxSequenceLength;
    }

    @Override
    public SequenceSchema newSchema(List<String> columnNames, List<ColumnMetaData> columnMetaData){
        return new SequenceSchema(columnNames,columnMetaData,minSequenceLength,maxSequenceLength);
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        int nCol = numColumns();

        int maxNameLength = 0;
        for(String s :getColumnNames()){
            maxNameLength = Math.max(maxNameLength,s.length());
        }

        //Header:
        sb.append("SequenceSchema(");

        if(minSequenceLength != null) sb.append("minSequenceLength=").append(minSequenceLength);
        if(maxSequenceLength != null){
            if(minSequenceLength != null) sb.append(",");
            sb.append("maxSequenceLength=").append(maxSequenceLength);
        }

        sb.append(")\n");
        sb.append(String.format("%-6s","idx")).append(String.format("%-"+(maxNameLength+8)+"s","name"))
                .append(String.format("%-15s","type")).append("meta data").append("\n");

        for( int i=0; i<nCol; i++ ){
            String colName = getName(i);
            ColumnType type = getType(i);
            ColumnMetaData meta = getMetaData(i);
            String paddedName = String.format("%-"+(maxNameLength+8)+"s","\"" + colName + "\"");
            sb.append(String.format("%-6d",i))
                    .append(paddedName)
                    .append(String.format("%-15s",type))
                    .append(meta).append("\n");
        }

        return sb.toString();
    }

    public static class Builder extends Schema.Builder {

        private Integer minSequenceLength;
        private Integer maxSequenceLength;

        public Builder minSequenceLength(int minSequenceLength){
            this.minSequenceLength = minSequenceLength;
            return this;
        }

        public Builder maxSequenceLength(int maxSequenceLength){
            this.maxSequenceLength = maxSequenceLength;
            return this;
        }


        @Override
        public SequenceSchema build(){
            return new SequenceSchema(this);
        }
    }
}
