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

package org.datavec.api.transform.analysis.quality;

import lombok.AllArgsConstructor;
import org.datavec.api.transform.ColumnType;
import org.datavec.api.transform.analysis.quality.bytes.BytesQualityAnalysisState;
import org.datavec.api.transform.analysis.quality.categorical.CategoricalQualityAnalysisState;
import org.datavec.api.transform.analysis.quality.integer.IntegerQualityAnalysisState;
import org.datavec.api.transform.analysis.quality.longq.LongQualityAnalysisState;
import org.datavec.api.transform.analysis.quality.real.RealQualityAnalysisState;
import org.datavec.api.transform.analysis.quality.string.StringQualityAnalysisState;
import org.datavec.api.transform.analysis.quality.time.TimeQualityAnalysisState;
import org.datavec.api.transform.metadata.*;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.function.BiFunction;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Add function used for undertaking quality analysis of a data set via Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class QualityAnalysisAddFunction
                implements BiFunction<List<QualityAnalysisState>, List<Writable>, List<QualityAnalysisState>>, Serializable {

    private Schema schema;

    @Override
    public List<QualityAnalysisState> apply(List<QualityAnalysisState> analysisStates, List<Writable> writables){
        if (analysisStates == null) {
            analysisStates = new ArrayList<>();
            List<ColumnType> columnTypes = schema.getColumnTypes();
            List<ColumnMetaData> columnMetaDatas = schema.getColumnMetaData();
            for (int i = 0; i < columnTypes.size(); i++) {
                switch (columnTypes.get(i)) {
                    case String:
                        analysisStates.add(new StringQualityAnalysisState((StringMetaData) columnMetaDatas.get(i)));
                        break;
                    case Integer:
                        analysisStates.add(new IntegerQualityAnalysisState((IntegerMetaData) columnMetaDatas.get(i)));
                        break;
                    case Long:
                        analysisStates.add(new LongQualityAnalysisState((LongMetaData) columnMetaDatas.get(i)));
                        break;
                    case Double:
                        analysisStates.add(new RealQualityAnalysisState((DoubleMetaData) columnMetaDatas.get(i)));
                        break;
                    case Categorical:
                        analysisStates.add(new CategoricalQualityAnalysisState((CategoricalMetaData) columnMetaDatas.get(i)));
                        break;
                    case Time:
                        analysisStates.add(new TimeQualityAnalysisState((TimeMetaData)columnMetaDatas.get(i)));
                        break;
                    case Bytes:
                        analysisStates.add(new BytesQualityAnalysisState()); //TODO
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown column type: " + columnTypes.get(i));
                }
            }
        }

        int size = analysisStates.size();
        if (size != writables.size())
            throw new IllegalStateException("Writables list and number of states does not match (" + writables.size()
                            + " vs " + size + ")");
        for (int i = 0; i < size; i++) {
            analysisStates.get(i).add(writables.get(i));
        }

        return analysisStates;
    }
}
