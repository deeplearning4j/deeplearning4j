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
package com.atilika.kuromoji;

import com.atilika.kuromoji.dict.Dictionary;
import com.atilika.kuromoji.viterbi.ViterbiNode.Type;

/**
 * Abstract token class with features shared by all tokens produced by all tokenizers
 */
public abstract class TokenBase {

    private static final int META_DATA_SIZE = 4;

    private final Dictionary dictionary;
    private final int wordId;
    private final String surface;
    private final int position;
    private final Type type;

    public TokenBase(int wordId, String surface, Type type, int position, Dictionary dictionary) {
        this.wordId = wordId;
        this.surface = surface;
        this.type = type;
        this.position = position;
        this.dictionary = dictionary;
    }

    /**
     * Gets the surface form of this token (表層形)
     *
     * @return surface form, not null
     */
    public String getSurface() {
        return surface;
    }

    /**
     * Predicate indicating whether this token is known (contained in the standard dictionary)
     *
     * @return true if the token is known, otherwise false
     */
    public boolean isKnown() {
        return type == Type.KNOWN;
    }

    /**
     * Predicate indicating whether this token is included is from the user dictionary
     * <p>
     * If a token is contained both in the user dictionary and standard dictionary, this method will return true
     *
     * @return true if this token is in user dictionary. false if not.
     */
    public boolean isUser() {
        return type == Type.USER;
    }

    /**
     * Gets the position/start index where this token is found in the input text
     *
     * @return token position
     */
    public int getPosition() {
        return position;
    }

    /**
     * Gets all features for this token as a comma-separated String
     *
     * @return token features, not null
     */
    public String getAllFeatures() {
        return dictionary.getAllFeatures(wordId);
    }

    /**
     * Gets all features for this token as a String array
     *
     * @return token feature array, not null
     */
    public String[] getAllFeaturesArray() {
        return dictionary.getAllFeaturesArray(wordId);
    }

    @Override
    public String toString() {
        return "Token{" + "surface='" + surface + '\'' + ", position=" + position + ", type=" + type + ", dictionary="
                        + dictionary + ", wordId=" + wordId + '}';
    }

    /**
     * Gets a numbered feature for this token
     *
     * @param feature  feature number
     * @return token feature, not null
     */
    protected String getFeature(int feature) {
        return dictionary.getFeature(wordId, feature - META_DATA_SIZE);
    }

}
