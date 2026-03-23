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

package org.eclipse.deeplearning4j.llm.eval;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import lombok.Builder;
import lombok.Data;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

@Data
@Builder
public class EvalResult {
    private String benchmarkName;
    private double primaryScore;
    private Map<String, Double> metricScores;
    private Map<String, Double> categoryScores;
    private int totalSamples;
    private int correctSamples;
    private long evaluationTimeMs;
    private List<SampleResult> sampleResults;

    public double accuracy() {
        return totalSamples > 0 ? (double) correctSamples / totalSamples : 0.0;
    }

    public void writeJson(File outputFile) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(outputFile, this);
    }

    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Benchmark: %s%n", benchmarkName));
        sb.append(String.format("Primary score: %.4f%n", primaryScore));
        sb.append(String.format("Samples: %d/%d correct (%.2f%%)%n",
                correctSamples, totalSamples, accuracy() * 100));
        sb.append(String.format("Time: %dms%n", evaluationTimeMs));
        if (metricScores != null && !metricScores.isEmpty()) {
            sb.append("Metrics:\n");
            for (Map.Entry<String, Double> entry : metricScores.entrySet()) {
                sb.append(String.format("  %s: %.4f%n", entry.getKey(), entry.getValue()));
            }
        }
        if (categoryScores != null && !categoryScores.isEmpty()) {
            sb.append("Categories:\n");
            for (Map.Entry<String, Double> entry : categoryScores.entrySet()) {
                sb.append(String.format("  %s: %.4f%n", entry.getKey(), entry.getValue()));
            }
        }
        return sb.toString();
    }
}
