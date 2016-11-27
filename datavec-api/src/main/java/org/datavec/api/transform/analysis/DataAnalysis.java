/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.api.transform.analysis;

import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.schema.Schema;
import lombok.AllArgsConstructor;
import lombok.Data;

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

        for( int i = 0; i < nCol; i++) {
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
