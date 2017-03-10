/*-*
 * Copyright © 2010-2015 Atilika Inc. and contributors (see CONTRIBUTORS.md)
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.  A copy of the
 * License is distributed with this work in the LICENSE.md file.  You may
 * also obtain a copy of the License from
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
