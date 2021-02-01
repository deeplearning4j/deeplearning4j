/*
 *  ******************************************************************************
 *  *
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

import com.atilika.kuromoji.util.StringUtils;

public class InsertedDictionary implements Dictionary {

    private static final String DEFAULT_FEATURE = "*";

    private static final String FEATURE_SEPARATOR = ",";

    private final String[] featuresArray;

    private final String featuresString;

    public InsertedDictionary(int features) {

        featuresArray = new String[features];

        for (int i = 0; i < features; i++) {
            featuresArray[i] = DEFAULT_FEATURE;
        }

        featuresString = StringUtils.join(featuresArray, FEATURE_SEPARATOR);
    }

    @Override
    public int getLeftId(int wordId) {
        return 0;
    }

    @Override
    public int getRightId(int wordId) {
        return 0;
    }

    @Override
    public int getWordCost(int wordId) {
        return 0;
    }

    @Override
    public String getAllFeatures(int wordId) {
        return featuresString;
    }

    @Override
    public String[] getAllFeaturesArray(int wordId) {
        return featuresArray;
    }

    @Override
    public String getFeature(int wordId, int... fields) {
        String[] features = new String[fields.length];

        for (int i = 0; i < features.length; i++) {
            features[i] = DEFAULT_FEATURE;
        }

        return StringUtils.join(features, FEATURE_SEPARATOR);
    }
}
