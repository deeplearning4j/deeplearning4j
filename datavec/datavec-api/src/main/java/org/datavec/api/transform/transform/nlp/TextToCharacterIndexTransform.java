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
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;

/**
 *
 * Convert each text value in a sequence to a longer sequence of integer indices.
 * For example, "abc" would be converted to [1, 2, 3]. Values in other columns will be duplicated.
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(callSuper = true, exclude = {""})
public class TextToCharacterIndexTransform extends BaseSequenceExpansionTransform {

    private Map<Character,Integer> characterIndexMap;
    private boolean exceptionOnUnknown;
    private transient Map<Character,List<Writable>> writableMap;

    /**
     *
     * @param columnName         Name of the text column
     * @param newColumnName      Name of the column after expansion
     * @param characterIndexMap  Character to integer index map
     * @param exceptionOnUnknown If true: throw an exception on unknown characters. False: skip unknown characters.
     */
    public TextToCharacterIndexTransform(@JsonProperty("columnName") String columnName,
                                         @JsonProperty("newColumnName") String newColumnName,
                                         @JsonProperty("characterIndexMap") Map<Character,Integer> characterIndexMap,
                                         @JsonProperty("exceptionOnUnknown") boolean exceptionOnUnknown){
        super(Collections.singletonList(columnName), Collections.singletonList(newColumnName));
        this.characterIndexMap = characterIndexMap;
        this.exceptionOnUnknown = exceptionOnUnknown;
    }

    @Override
    protected List<ColumnMetaData> expandedColumnMetaDatas(List<ColumnMetaData> origColumnMeta, List<String> expandedColumnNames) {
        return Collections.<ColumnMetaData>singletonList(new IntegerMetaData(expandedColumnNames.get(0), 0, characterIndexMap.size()-1));
    }

    @Override
    protected List<List<Writable>> expandTimeStep(List<Writable> currentStepValues) {
        if(writableMap == null){
            Map<Character,List<Writable>> m = new HashMap<>();
            for(Map.Entry<Character,Integer> entry : characterIndexMap.entrySet()){
                m.put(entry.getKey(), Collections.<Writable>singletonList(new IntWritable(entry.getValue())));
            }
            writableMap = m;
        }
        List<List<Writable>> out = new ArrayList<>();
        char[] cArr = currentStepValues.get(0).toString().toCharArray();
        for( char c : cArr ){
            List<Writable> w = writableMap.get(c);
            if(w == null ){
                if(exceptionOnUnknown){
                    throw new IllegalStateException("Unknown character found in text: \"" + c + "\"");
                }
                continue;
            }

            out.add(w);
        }

        return out;
    }
}
