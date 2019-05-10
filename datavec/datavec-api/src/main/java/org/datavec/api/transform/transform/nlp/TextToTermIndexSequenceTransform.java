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

package org.datavec.api.transform.transform.nlp;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.metadata.ColumnMetaData;
import org.datavec.api.transform.metadata.IntegerMetaData;
import org.datavec.api.transform.sequence.expansion.BaseSequenceExpansionTransform;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;

/**
 *
 * Convert each text value in a sequence to a longer sequence of integer indices.
 * For example, "zero one two" would be converted to [0, 1, 2]. Values in other
 * columns will be duplicated.
 *
 * @author Dave Kale
 */
@Data
@EqualsAndHashCode(callSuper = true, exclude = {"writableMap"})
@JsonIgnoreProperties({"writableMap"})
@JsonInclude(JsonInclude.Include.NON_NULL)
public class TextToTermIndexSequenceTransform extends BaseSequenceExpansionTransform {

    private Map<String,Integer> wordIndexMap;
    private String delimiter;
    private boolean exceptionOnUnknown;
    private transient Map<String,List<Writable>> writableMap;

    /**
     *
     * @param columnName         Name of the text column
     * @param newColumnName      Name of the column after expansion
     * @param vocabulary         Vocabulary
     * @param delimiter          Delimiter
     * @param exceptionOnUnknown If true: throw an exception on unknown characters. False: skip unknown characters.
     */
    public TextToTermIndexSequenceTransform(String columnName, String newColumnName, List<String> vocabulary,
                                            String delimiter, boolean exceptionOnUnknown){
        super(Collections.singletonList(columnName), Collections.singletonList(newColumnName));
        this.wordIndexMap = new HashMap<>();
        for(int i = 0; i < vocabulary.size(); ++i) {
            this.wordIndexMap.put(vocabulary.get(i), i);
        }
        this.delimiter = delimiter;
        this.exceptionOnUnknown = exceptionOnUnknown;
    }

    /**
     *
     * @param columnName         Name of the text column
     * @param newColumnName      Name of the column after expansion
     * @param wordIndexMap       Map from terms in vocabulary to indeces
     * @param delimiter          Delimiter
     * @param exceptionOnUnknown If true: throw an exception on unknown characters. False: skip unknown characters.
     */
    public TextToTermIndexSequenceTransform(@JsonProperty("columnName") String columnName,
                                            @JsonProperty("newColumnName") String newColumnName,
                                            @JsonProperty("wordIndexMap") Map<String,Integer> wordIndexMap,
                                            @JsonProperty("delimiter") String delimiter,
                                            @JsonProperty("exceptionOnUnknown") boolean exceptionOnUnknown){
        super(Collections.singletonList(columnName), Collections.singletonList(newColumnName));
        this.wordIndexMap = wordIndexMap;
        this.delimiter = delimiter;
        this.exceptionOnUnknown = exceptionOnUnknown;
    }

    @Override
    protected List<ColumnMetaData> expandedColumnMetaDatas(List<ColumnMetaData> origColumnMeta, List<String> expandedColumnNames) {
        return Collections.<ColumnMetaData>singletonList(new IntegerMetaData(expandedColumnNames.get(0), 0, wordIndexMap.size()-1));
    }

    @Override
    protected List<List<Writable>> expandTimeStep(List<Writable> currentStepValues) {
        if(writableMap == null){
            Map<String,List<Writable>> m = new HashMap<>();
            for(Map.Entry<String,Integer> entry : wordIndexMap.entrySet()) {
                m.put(entry.getKey(), Collections.<Writable>singletonList(new IntWritable(entry.getValue())));
            }
            writableMap = m;
        }
        List<List<Writable>> out = new ArrayList<>();
        String text = currentStepValues.get(0).toString();
        String[] tokens = text.split(this.delimiter);
        for(String token : tokens ){
            List<Writable> w = writableMap.get(token);
            if(w == null) {
                if(exceptionOnUnknown) {
                    throw new IllegalStateException("Unknown token found in text: \"" + token + "\"");
                }
                continue;
            }
            out.add(w);
        }

        return out;
    }
}