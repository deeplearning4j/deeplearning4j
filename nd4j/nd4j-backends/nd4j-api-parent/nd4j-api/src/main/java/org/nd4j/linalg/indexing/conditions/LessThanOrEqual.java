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
 * Created by agibsonccc on 10/8/14.
 */
public class LessThanOrEqual extends BaseCondition {

    /**
     * Special constructor for pairwise boolean operations.
     */
    public LessThanOrEqual() {
        super(0.0);
    }

    public LessThanOrEqual(Number value) {
        super(value);
    }

    /**
     * Returns condition ID for native side
     *
     * @return
     */
    @Override
    public int condtionNum() {
        return 4;
    }

    @Override
    public Boolean apply(Number input) {
        return input.floatValue() <= value.doubleValue();
    }
}
