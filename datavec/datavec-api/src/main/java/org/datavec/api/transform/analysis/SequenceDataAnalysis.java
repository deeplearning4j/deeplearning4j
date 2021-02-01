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

package org.datavec.api.transform.analysis;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.analysis.columns.ColumnAnalysis;
import org.datavec.api.transform.analysis.sequence.SequenceLengthAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.serde.JsonMappers;
import org.datavec.api.transform.serde.JsonSerializer;
import org.datavec.api.transform.serde.YamlSerializer;
import org.nd4j.shade.jackson.databind.exc.InvalidTypeIdException;

import java.io.IOException;
import java.util.List;

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
        } catch (InvalidTypeIdException e){
            if(e.getMessage().contains("@class")){
                try{
                    //JSON may be legacy (1.0.0-alpha or earlier), attempt to load it using old format
                    return JsonMappers.getLegacyMapper().readValue(json, SequenceDataAnalysis.class);
                } catch (IOException e2){
                    throw new RuntimeException(e2);
                }
            }
            throw new RuntimeException(e);
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
