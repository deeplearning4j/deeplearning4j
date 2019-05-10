/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.transform.ndarray;

import lombok.Data;
import lombok.NonNull;
import org.datavec.api.transform.Distance;
import org.datavec.api.transform.Transform;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.DoubleMetaData;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Calculate the distance (cosine similarity, EUCLIDEAN, MANHATTAN) between two INDArrays
 *
 * @author Alex Black
 */
@Data
public class NDArrayDistanceTransform implements Transform {

    private String newColumnName;
    private Distance distance;
    private String firstCol;
    private String secondCol;

    private Schema inputSchema;

    public NDArrayDistanceTransform(@JsonProperty("newColumnName") @NonNull String newColumnName,
                    @JsonProperty("distance") @NonNull Distance distance,
                    @JsonProperty("firstCol") @NonNull String firstCol,
                    @JsonProperty("secondCol") @NonNull String secondCol) {
        this.newColumnName = newColumnName;
        this.distance = distance;
        this.firstCol = firstCol;
        this.secondCol = secondCol;
    }


    @Override
    public String toString() {
        return "NDArrayDistanceTransform(newColumnName=\"" + newColumnName + "\",distance=" + distance + ",firstCol="
                        + firstCol + ",secondCol=" + secondCol + ")";
    }

    @Override
    public void setInputSchema(Schema inputSchema) {
        if (!inputSchema.hasColumn(firstCol)) {
            throw new IllegalStateException("Input schema does not have first column: " + firstCol);
        }
        if (!inputSchema.hasColumn(secondCol)) {
            throw new IllegalStateException("Input schema does not have first column: " + secondCol);
        }

        this.inputSchema = inputSchema;
    }

    @Override
    public List<Writable> map(List<Writable> writables) {
        int idxFirst = inputSchema.getIndexOfColumn(firstCol);
        int idxSecond = inputSchema.getIndexOfColumn(secondCol);

        INDArray arr1 = ((NDArrayWritable) writables.get(idxFirst)).get();
        INDArray arr2 = ((NDArrayWritable) writables.get(idxSecond)).get();

        double d;
        switch (distance) {
            case COSINE:
                d = Transforms.cosineSim(arr1, arr2);
                break;
            case EUCLIDEAN:
                d = Transforms.euclideanDistance(arr1, arr2);
                break;
            case MANHATTAN:
                d = Transforms.manhattanDistance(arr1, arr2);
                break;
            default:
                throw new UnsupportedOperationException("Unknown or not supported distance metric: " + distance);
        }

        List<Writable> out = new ArrayList<>(writables.size() + 1);
        out.addAll(writables);
        out.add(new DoubleWritable(d));

        return out;
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        List<List<Writable>> out = new ArrayList<>();
        for (List<Writable> l : sequence) {
            out.add(map(l));
        }
        return out;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Object mapSequence(Object sequence) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Schema transform(Schema inputSchema) {
        List<ColumnMetaData> newMeta = new ArrayList<>(inputSchema.getColumnMetaData());
        newMeta.add(new DoubleMetaData(newColumnName));
        return inputSchema.newSchema(newMeta);
    }

    @Override
    public String outputColumnName() {
        return newColumnName;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[] {outputColumnName()};
    }

    @Override
    public String[] columnNames() {
        return new String[] {firstCol, secondCol};
    }

    @Override
    public String columnName() {
        return columnNames()[0];
    }
}
