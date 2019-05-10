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

package org.deeplearning4j.gym.test;

import org.json.JSONObject;
import org.mockito.ArgumentMatcher;

import static org.mockito.Matchers.argThat;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/11/16.
 */


public class JSONObjectMatcher implements ArgumentMatcher<JSONObject> {
    private final JSONObject expected;

    public JSONObjectMatcher(JSONObject expected) {
        this.expected = expected;
    }

    public static JSONObject jsonEq(JSONObject expected) {
        return argThat(new JSONObjectMatcher(expected));
    }


    @Override
    public boolean matches(JSONObject argument) {
        if (expected == null)
            return argument == null;
        return expected.toString().equals(argument.toString());    }
}
