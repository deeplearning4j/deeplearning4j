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

package org.datavec.api.transform.analysis;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.sequence.SequenceLengthAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.JsonSerializer;
import org.datavec.api.transform.serde.YamlSerializer;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.List;

/**
 * Created by Alex on 12/03/2016.
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class SequenceDataAnalysis extends DataAnalysis {

    private SequenceLengthAnalysis sequenceLengthAnalysis;

    public SequenceDataAnalysis(Schema schema, List<ColumnAnalysis> columnAnalysis,
                    SequenceLengthAnalysis sequenceAnalysis) {
        super(schema, columnAnalysis);
        this.sequenceLengthAnalysis = sequenceAnalysis;
    }

    protected SequenceDataAnalysis(){
        //No arg for JSON
    }

    public static SequenceDataAnalysis fromJson(String json){
        try{
            return new JsonSerializer().getObjectMapper().readValue(json, SequenceDataAnalysis.class);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    public static SequenceDataAnalysis fromYaml(String yaml){
        try{
            return new YamlSerializer().getObjectMapper().readValue(yaml, SequenceDataAnalysis.class);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return sequenceLengthAnalysis + "\n" + super.toString();
    }
}
