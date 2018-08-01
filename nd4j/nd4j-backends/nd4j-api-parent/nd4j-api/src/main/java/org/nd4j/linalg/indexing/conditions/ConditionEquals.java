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

package org.nd4j.linalg.indexing.conditions;

/**
 * Returns true when all of the conditions equal each other
 *
 * @author Adam Gibson
 */
public class ConditionEquals implements Condition {

    private Condition[] conditions;

    public ConditionEquals(Condition... conditions) {
        this.conditions = conditions;
    }

    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public int condtionNum() {
        return -1;
    }

    @Override
    public double getValue() {
        return -1;
    }

    @Override
    public double epsThreshold() {
        return 0;
    }

    @Override
    public Boolean apply(Number input) {
        boolean ret = conditions[0].apply(input);
        for (int i = 1; i < conditions.length; i++) {
            ret = ret == conditions[i].apply(input);
        }
        return ret;
    }
}
