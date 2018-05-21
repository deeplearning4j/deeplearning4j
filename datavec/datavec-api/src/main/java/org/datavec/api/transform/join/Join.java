/*-
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

import lombok.Data;
import org.apache.commons.lang3.ArrayUtils;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.NullWritable;
import org.datavec.api.writable.Writable;

import java.io.Serializable;
import java.util.*;

/**
 * Join class: used to specify a join (like an SQL join)
 *
 * @author Alex Black
 */
@Data
public class Join implements Serializable {

    /**
     * Type of join<br>
     * Inner: Return examples where the join column values occur in both
     * LeftOuter: Return all examples from left data, whether there is a matching right value or not.
     * (If not: right values will have NullWritable instead)<br>
     * RightOuter: Return all examples from the right data, whether there is a matching left value or not.
     * (If not: left values will have NullWritable instead)<br>
     * FullOuter: return all examples from both left/right, whether there is a matching value from the other side or not.
     * (If not: other values will have NullWritable instead)
     */
    public enum JoinType {
        Inner, LeftOuter, RightOuter, FullOuter
    };

    private JoinType joinType;
    private Schema leftSchema;
    private Schema rightSchema;
    private String[] joinColumnsLeft;
    private String[] joinColumnsRight;


    private Join(Builder builder) {
        this.joinType = builder.joinType;
        this.leftSchema = builder.leftSchema;
        this.rightSchema = builder.rightSchema;
        this.joinColumnsLeft = builder.joinColumnsLeft;
        this.joinColumnsRight = builder.joinColumnsRight;

        //Perform validation: ensure columns are correct, etc
        if (joinType == null)
            throw new IllegalArgumentException("Join type cannot be null");
        if (leftSchema == null)
            throw new IllegalArgumentException("Left schema cannot be null");
        if (rightSchema == null)
            throw new IllegalArgumentException("Right schema cannot be null");
        if (joinColumnsLeft == null || joinColumnsLeft.length == 0) {
            throw new IllegalArgumentException("Invalid left join columns: "
                            + (joinColumnsLeft == null ? null : Arrays.toString(joinColumnsLeft)));
        }
        if (joinColumnsRight == null || joinColumnsRight.length == 0) {
            throw new IllegalArgumentException("Invalid right join columns: "
                            + (joinColumnsRight == null ? null : Arrays.toString(joinColumnsRight)));
        }

        //Check that the join columns actually appear in the schemas:
        for (String leftCol : joinColumnsLeft) {
            if (!leftSchema.hasColumn(leftCol)) {
                throw new IllegalArgumentException("Cannot perform join: left join column \"" + leftCol
                                + "\" does not exist in left schema. All columns in left schema: " + leftSchema.getColumnNames());
            }
        }

        for (String rightCol : joinColumnsRight) {
            if (!rightSchema.hasColumn(rightCol)) {
                throw new IllegalArgumentException("Cannot perform join: right join column \"" + rightCol
                                + "\" does not exist in right schema. All columns in right schema: " + rightSchema.getColumnNames());
            }
        }
    }


    public static class Builder {

        private JoinType joinType;
        private Schema leftSchema;
        private Schema rightSchema;
        private String[] joinColumnsLeft;
        private String[] joinColumnsRight;

        public Builder(JoinType type) {
            this.joinType = type;
        }

        public Builder setSchemas(Schema left, Schema right) {
            this.leftSchema = left;
            this.rightSchema = right;
            return this;
        }


        /**
         * @deprecated Use {@link #setJoinColumns(String...)}
         */
        @Deprecated
        public Builder setKeyColumns(String... keyColumnNames) {
            return setJoinColumns(keyColumnNames);
        }

        /**
         * @deprecated Use {@link #setJoinColumnsLeft(String...)}
         */
        @Deprecated
        public Builder setKeyColumnsLeft(String... keyColumnNames) {
            return setJoinColumnsLeft(keyColumnNames);
        }

        /**
         * @deprecated Use {@link #setJoinColumnsRight(String...)}
         */
        @Deprecated
        public Builder setKeyColumnsRight(String... keyColumnNames) {
            return setJoinColumnsRight(keyColumnNames);
        }

        /** Specify the column(s) to join on.
         * Here, we are assuming that both data sources have the same column names. If this is not the case,
         * use {@link #setJoinColumnsLeft(String...)} and {@link #setJoinColumnsRight(String...)}.
         * The idea: join examples where firstDataValues(joinColumNames[i]) == secondDataValues(joinColumnNames[i]) for all i
         * @param joinColumnNames    Name of the columns to use as the key to join on
         */
        public Builder setJoinColumns(String... joinColumnNames) {
            setJoinColumnsLeft(joinColumnNames);
            return setJoinColumnsRight(joinColumnNames);
        }

        /**
         * Specify the names of the columns to join on, for the left data)
         * The idea: join examples where firstDataValues(joinColumNamesLeft[i]) == secondDataValues(joinColumnNamesRight[i]) for all i
         * @param joinColumnNames Names of the columns to join on (for left data)
         */
        public Builder setJoinColumnsLeft(String... joinColumnNames) {
            this.joinColumnsLeft = joinColumnNames;
            return this;
        }

        /**
         * Specify the names of the columns to join on, for the right data)
         * The idea: join examples where firstDataValues(joinColumNamesLeft[i]) == secondDataValues(joinColumnNamesRight[i]) for all i
         * @param joinColumnNames Names of the columns to join on (for left data)
         */
        public Builder setJoinColumnsRight(String... joinColumnNames) {
            this.joinColumnsRight = joinColumnNames;
            return this;
        }

        public Join build() {
            if (leftSchema == null || rightSchema == null)
                throw new IllegalStateException("Cannot build Join: left and/or right schemas are null");
            return new Join(this);
        }
    }



    public Schema getOutputSchema() {
        if (leftSchema == null)
            throw new IllegalStateException("Left schema is not set (null)");
        if (rightSchema == null)
            throw new IllegalStateException("Right schema is not set (null)");
        if (joinColumnsLeft == null)
            throw new IllegalStateException("Left key columns are not set (null)");
        if (joinColumnsRight == null)
            throw new IllegalArgumentException("Right key columns are not set (null");

        //Approach here: take the left schema, plus the right schema (excluding the key columns from the right schema)
        List<ColumnMetaData> metaDataOut = new ArrayList<>(leftSchema.getColumnMetaData());

        Set<String> keySetRight = new HashSet<>();
        Collections.addAll(keySetRight, joinColumnsRight);

        for (ColumnMetaData rightMeta : rightSchema.getColumnMetaData()) {
            if (keySetRight.contains(rightMeta.getName()))
                continue;;
            metaDataOut.add(rightMeta);
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
    public List<Writable> joinExamples(List<Writable> leftExample, List<Writable> rightExample) {

        List<Writable> out = new ArrayList<>();
        if (leftExample == null) {
            if (rightExample == null)
                throw new IllegalArgumentException(
                                "Cannot join examples: Both examples are null (max 1 allowed to be null)");

            //Insert a set of null writables...
            //Complication here: the **key values** should still exist (we have to extract them from second value)
            int nLeft = leftSchema.numColumns();
            List<String> leftNames = leftSchema.getColumnNames();
            int keysSoFar = 0;
            for (int i = 0; i < nLeft; i++) {
                String name = leftNames.get(i);
                if (ArrayUtils.contains(joinColumnsLeft, name)) {
                    //This would normally be where the left key came from...
                    //So let's get the key value from the *right* example
                    String rightKeyName = joinColumnsRight[keysSoFar];
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
        if (rightExample == null) {
            //Insert a set of null writables...
            int nRight = rightSchema.numColumns();
            for (int i = 0; i < nRight; i++) {
                String name = rightNames.get(i);
                if (ArrayUtils.contains(joinColumnsRight, name))
                    continue; //Skip the key column value
                out.add(NullWritable.INSTANCE);
            }
        } else {
            //Add all values from right, except for key columns...
            for (int i = 0; i < rightExample.size(); i++) {
                String name = rightNames.get(i);
                if (ArrayUtils.contains(joinColumnsRight, name))
                    continue; //Skip the key column value
                out.add(rightExample.get(i));
            }
        }

        return out;
    }
}
