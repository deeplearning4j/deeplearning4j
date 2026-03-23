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

package org.eclipse.deeplearning4j.llm.eval.dataset;

import java.util.*;

/**
 * User-defined evaluation dataset from an explicit list of samples or builder pattern.
 */
public class CustomDataset implements EvalDataset {

    private final String datasetName;
    private final List<EvalSample> samples;

    public CustomDataset(String name, List<EvalSample> samples) {
        this.datasetName = name;
        this.samples = new ArrayList<>(samples);
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    @Override
    public String name() {
        return datasetName;
    }

    @Override
    public int size() {
        return samples.size();
    }

    @Override
    public EvalSample get(int index) {
        return samples.get(index);
    }

    @Override
    public List<EvalSample> getSplit(String split) {
        if ("test".equals(split)) return Collections.unmodifiableList(samples);
        return Collections.emptyList();
    }

    @Override
    public Iterator<EvalSample> iterator() {
        return samples.iterator();
    }

    public static class Builder {
        private final String name;
        private final List<EvalSample> samples = new ArrayList<>();

        private Builder(String name) {
            this.name = name;
        }

        public Builder addSample(EvalSample sample) {
            samples.add(sample);
            return this;
        }

        public Builder addSample(String id, String input, String... references) {
            samples.add(EvalSample.builder()
                    .id(id)
                    .input(input)
                    .references(Arrays.asList(references))
                    .build());
            return this;
        }

        public Builder addMultipleChoice(String id, String input, List<String> choices, String correctAnswer) {
            samples.add(EvalSample.builder()
                    .id(id)
                    .input(input)
                    .choices(choices)
                    .references(List.of(correctAnswer))
                    .build());
            return this;
        }

        public CustomDataset build() {
            return new CustomDataset(name, samples);
        }
    }
}
