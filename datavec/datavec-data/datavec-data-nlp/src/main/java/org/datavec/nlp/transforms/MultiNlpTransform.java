/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.NDArrayMetaData;
import org.datavec.api.transform.transform.BaseColumnTransform;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.list.NDArrayList;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collections;
import java.util.List;

/**
 * A multi NLP transform takes in 1 or more bag of words transforms as a pipeline
 * and runs them in sequence.
 * This transform takes in a column name and 1 or more bag of words transforms to run.
 * Lastly, a new column name is specified.
 *
 * @author Adam Gibson
 */
public class MultiNlpTransform extends BaseColumnTransform implements BagOfWordsTransform {

    private BagOfWordsTransform[] transforms;
    private String newColumnName;
    private List<String> vocabWords;

    /**
     *
     * @param columnName
     * @param transforms
     * @param newColumnName
     */
    @JsonCreator
    public MultiNlpTransform(@JsonProperty("columnName") String columnName,
                             @JsonProperty("transforms") BagOfWordsTransform[] transforms,
                             @JsonProperty("newColumnName") String newColumnName) {
        super(columnName);
        this.transforms = transforms;
        this.vocabWords = transforms[0].vocabWords();
        if(transforms.length > 1) {
            for(int i = 1; i < transforms.length; i++) {
                if(!transforms[i].vocabWords().equals(vocabWords)) {
                    throw new IllegalArgumentException("Vocab words not consistent across transforms!");
                }
            }
        }

        this.newColumnName = newColumnName;
    }

    @Override
    public Object mapSequence(Object sequence) {
        NDArrayList ndArrayList = new NDArrayList();
        for(BagOfWordsTransform bagofWordsTransform : transforms) {
            ndArrayList.addAll(new NDArrayList(bagofWordsTransform.transformFromObject((List<List<Object>>) sequence)));
        }

        return ndArrayList.array();
    }

    @Override
    public List<List<Writable>> mapSequence(List<List<Writable>> sequence) {
     return Collections.singletonList(Collections.<Writable>singletonList(new NDArrayWritable(transformFrom(sequence))));
    }

    @Override
    public ColumnMetaData getNewColumnMetaData(String newName, ColumnMetaData oldColumnType) {
        return new NDArrayMetaData(newName,outputShape());
    }

    @Override
    public Writable map(Writable columnWritable) {
        throw new UnsupportedOperationException("Only able to add for time series");
    }

    @Override
    public String toString() {
        return newColumnName;
    }

    @Override
    public Object map(Object input) {
        throw new UnsupportedOperationException("Only able to add for time series");
    }

    @Override
    public long[] outputShape() {
        long[] ret = new long[transforms[0].outputShape().length];
        int validatedRank = transforms[0].outputShape().length;
        for(int i = 1; i < transforms.length; i++) {
            if(transforms[i].outputShape().length != validatedRank) {
                throw new IllegalArgumentException("Inconsistent shape length at transform " + i + " , should have been: " + validatedRank);
            }
        }
        for(int i = 0; i < transforms.length; i++) {
            for(int j = 0; j < validatedRank; j++)
            ret[j] += transforms[i].outputShape()[j];
        }

        return ret;
    }

    @Override
    public List<String> vocabWords() {
        return vocabWords;
    }

    @Override
    public INDArray transformFromObject(List<List<Object>> tokens) {
        NDArrayList ndArrayList = new NDArrayList();
        for(BagOfWordsTransform bagofWordsTransform : transforms) {
            INDArray arr2 = bagofWordsTransform.transformFromObject(tokens);
            arr2 = arr2.reshape(arr2.length());
            NDArrayList newList = new NDArrayList(arr2,(int) arr2.length());
            ndArrayList.addAll(newList);        }

        return ndArrayList.array();
    }

    @Override
    public INDArray transformFrom(List<List<Writable>> tokens) {
        NDArrayList ndArrayList = new NDArrayList();
        for(BagOfWordsTransform bagofWordsTransform : transforms) {
            INDArray arr2 = bagofWordsTransform.transformFrom(tokens);
            arr2 = arr2.reshape(arr2.length());
            NDArrayList newList = new NDArrayList(arr2,(int) arr2.length());
            ndArrayList.addAll(newList);
        }

        return ndArrayList.array();
    }


}
