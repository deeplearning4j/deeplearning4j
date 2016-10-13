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

package org.datavec.api.transform.join;

import org.datavec.api.writable.NullWritable;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import lombok.Data;
import org.apache.commons.lang3.ArrayUtils;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by Alex on 18/04/2016.
 */
@Data
public class Join implements Serializable {

    /**
     * Type of join
     */
    public enum JoinType {Inner, LeftOuter, RightOuter, FullOuter};

    private JoinType joinType;
    private Schema leftSchema;
    private Schema rightSchema;
    private String[] keyColumnsLeft;
    private String[] keyColumnsRight;


    private Join(Builder builder){
        this.joinType = builder.joinType;
        this.leftSchema = builder.leftSchema;
        this.rightSchema = builder.rightSchema;
        this.keyColumnsLeft = builder.keyColumnsLeft;
        this.keyColumnsRight = builder.keyColumnsRight;
    }


    public static class Builder {

        private JoinType joinType;
        private Schema leftSchema;
        private Schema rightSchema;
        private String[] keyColumnsLeft;
        private String[] keyColumnsRight;

        public Builder(JoinType type){
            this.joinType = type;
        }

        public Builder setSchemas(Schema left, Schema right){
            this.leftSchema = left;
            this.rightSchema = right;
            return this;
        }

        /** Specify the columns to use as the key. (With multiple columns: compound key)
         * Here, we are assuming that both data sources have the same key column names. If this is not the case,
         * use {@link #setKeyColumnsLeft(String...)} and {@link #setKeyColumnsRight(String...)}.
         * @param keyColumnNames    Name of the columns to use as the key to join on
         */
        public Builder setKeyColumns(String... keyColumnNames){
            setKeyColumnsLeft(keyColumnNames);
            return setKeyColumnsRight(keyColumnNames);
        }

        public Builder setKeyColumnsLeft(String... keyColumnNames){
            this.keyColumnsLeft = keyColumnNames;
            return this;
        }

        public Builder setKeyColumnsRight(String... keyColumnNames){
            this.keyColumnsRight = keyColumnNames;
            return this;
        }

        public Join build(){
            if(leftSchema == null || rightSchema == null) throw new IllegalStateException("Cannot build Join: left and/or right schemas are null");
            return new Join(this);
        }
    }



    public Schema getOutputSchema(){
        if(leftSchema == null) throw new IllegalStateException("Left schema is not set (null)");
        if(rightSchema == null) throw new IllegalStateException("Right schema is not set (null)");
        if(keyColumnsLeft == null) throw new IllegalStateException("Left key columns are not set (null)");
        if(keyColumnsRight == null) throw new IllegalArgumentException("Right key columns are not set (null");

        List<ColumnMetaData> metaDataOut = new ArrayList<>();
        Set<String> columnNamesSeenSoFar = new HashSet<>();

        List<String> columnNamesLeft = leftSchema.getColumnNames();
        List<String> columnNamesRight = rightSchema.getColumnNames();
        List<ColumnMetaData> metaDataLeft = leftSchema.getColumnMetaData();
        List<ColumnMetaData> metaDataRight = rightSchema.getColumnMetaData();

        for( int i=0; i<columnNamesLeft.size(); i++ ){
            String name = columnNamesLeft.get(i);
            metaDataOut.add(metaDataLeft.get(i));

            columnNamesSeenSoFar.add(name);
        }

        for( int i=0; i<columnNamesRight.size(); i++ ){
            String name = columnNamesRight.get(i);
            if(ArrayUtils.contains(keyColumnsRight,name)) continue; //Skip the right key column
            if(columnNamesSeenSoFar.contains(name)){
                throw new IllegalStateException("Cannot produce output schema: columns with name \"" + name +
                        "\" appear in both left and right schemas (and is not a key column for right schema)");
            }
            metaDataOut.add(metaDataRight.get(i));
        }

        return leftSchema.newSchema(metaDataOut);
    }

    /**
     * Join the examples.
     * Note: left or right examples may be null; if they are null, the appropriate NullWritables are inserted.
     *
     * @param leftExample
     * @param rightExample
     * @return
     */
    public List<Writable> joinExamples(List<Writable> leftExample, List<Writable> rightExample){

        List<Writable> out = new ArrayList<>();
        if(leftExample == null){
            if(rightExample == null) throw new IllegalArgumentException("Cannot join examples: Both examples are null (max 1 allowed to be null)");

            //Insert a set of null writables...
            //Complication here: the **key values** should still exist (we have to extract them from second value)
            int nLeft = leftSchema.numColumns();
            List<String> leftNames = leftSchema.getColumnNames();
            int keysSoFar = 0;
            for( int i=0; i<nLeft; i++ ){
                String name = leftNames.get(i);
                if(ArrayUtils.contains(keyColumnsLeft,name)){
                    //This would normally be where the left key came from...
                    //So let's get the key value from the *right* example
                    String rightKeyName = keyColumnsRight[keysSoFar];
                    int idxOfRightKey = rightSchema.getIndexOfColumn(rightKeyName);
                    out.add(rightExample.get(idxOfRightKey));
                } else {
                    //Not a key column, so just add a NullWritable
                    out.add(NullWritable.INSTANCE);
                }
            }
        } else {
            out.addAll(leftExample);
        }

        List<String> rightNames = rightSchema.getColumnNames();
        if(rightExample == null){
            //Insert a set of null writables...
            int nRight = rightSchema.numColumns();
            for( int i=0; i<nRight; i++ ){
                String name = rightNames.get(i);
                if(ArrayUtils.contains(keyColumnsRight,name)) continue; //Skip the key column value
                out.add(NullWritable.INSTANCE);
            }
        } else {
            //Add all values from right, except for key columns...
            for(int i=0; i<rightExample.size(); i++){
                String name = rightNames.get(i);
                if(ArrayUtils.contains(keyColumnsRight,name)) continue; //Skip the key column value
                out.add(rightExample.get(i));
            }
        }

        return out;
    }
}
