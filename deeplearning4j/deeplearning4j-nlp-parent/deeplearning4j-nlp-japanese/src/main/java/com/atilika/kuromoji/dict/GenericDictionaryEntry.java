/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
package com.atilika.kuromoji.dict;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class GenericDictionaryEntry extends DictionaryEntryBase implements Serializable {

    private final List<String> posFeatures;
    private final List<String> features;

    public GenericDictionaryEntry(Builder builder) {
        super(builder.surface, builder.leftId, builder.rightId, builder.wordCost);
        posFeatures = builder.pos;
        features = builder.features;
    }

    public List<String> getPosFeatures() {
        return posFeatures;
    }

    public List<String> getFeatures() {
        return features;
    }

    public static class Builder {
        private String surface;
        private short leftId;
        private short rightId;
        private short wordCost;
        private List<String> pos = new ArrayList<>();
        private List<String> features = new ArrayList<>();

        public Builder surface(String surface) {
            this.surface = surface;
            return this;
        }

        public Builder leftId(short leftId) {
            this.leftId = leftId;
            return this;
        }

        public Builder rightId(short rightId) {
            this.rightId = rightId;
            return this;
        }

        public Builder wordCost(short wordCost) {
            this.wordCost = wordCost;
            return this;
        }

        public Builder pos(List<String> pos) {
            this.pos = pos;
            return this;
        }

        public Builder features(List<String> features) {
            this.features = features;
            return this;
        }

        public GenericDictionaryEntry build() {
            return new GenericDictionaryEntry(this);
        }
    }
}
