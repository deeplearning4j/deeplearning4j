/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.datavec.nlp.transforms;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Data
@EqualsAndHashCode(callSuper = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties({"gazeteer"})
public class GazeteerTransform extends BaseColumnTransform implements BagOfWordsTransform {

    private String newColumnName;
    private List<String> wordList;
    private Set<String> gazeteer;

    @JsonCreator
    public GazeteerTransform(@JsonProperty("columnName") String columnName,
                             @JsonProperty("newColumnName")String newColumnName,
                             @JsonProperty("wordList") List<String> wordList) {
        super(columnName);
        this.newColumnName = newColumnName;
        this.wordList = wordList;
        this.gazeteer = new HashSet<>(wordList);
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new NDArrayMetaData(newName,new long[]{wordList.size()});
    }

    @Override
    public Writable map(Writable columnWritable) {
       throw new UnsupportedOperationException();
    }

    @Override
    public Object mapSequence(Object sequence) {
        List<List<Object>> sequenceInput = (List<List<Object>>) sequence;
        INDArray ret = Nd4j.create(DataType.FLOAT, wordList.size());

        for(List<Object> list : sequenceInput) {
            for(Object token : list) {
                String s = token.toString();
                if(gazeteer.contains(s)) {
                    ret.putScalar(wordList.indexOf(s),1);
                }
            }
        }
        return ret;
    }



    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
        INDArray arr = (INDArray) mapSequence((Object) sequence);
        return Collections.singletonList(Collections.<Writable>singletonList(new NDArrayWritable(arr)));
    }

    @Override
    public String toString() {
        return newColumnName;
    }

    @Override
    public Object map(Object input) {
        return gazeteer.contains(input.toString());
    }

    @Override
    public String outputColumnName() {
        return newColumnName;
    }

    @Override
    public String[] outputColumnNames() {
        return new String[]{newColumnName};
    }

    @Override
    public String[] columnNames() {
        return new String[]{columnName()};
    }

    @Override
    public String columnName() {
        return columnName;
    }

    @Override
    public long[] outputShape() {
        return new long[]{wordList.size()};
    }

    @Override
    public List<String> vocabWords() {
        return wordList;
    }

    @Override
    public INDArray transformFromObject(List<List<Object>> tokens) {
        return (INDArray) mapSequence(tokens);
    }

    @Override
    public INDArray transformFrom(List<List<Writable>> tokens) {
        return (INDArray) mapSequence((Object) tokens);
    }
}
