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

package org.deeplearning4j.arbiter.util;

import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 29/06/2017.
 */
public class LeafUtils {

    private LeafUtils() {}

    /**
     * Returns a list of unique objects, not using the .equals() method, but rather using ==
     *
     * @param allLeaves Leaf values to process
     * @return A list of unique parameter space values
     */
    public static List<ParameterSpace> getUniqueObjects(List<ParameterSpace> allLeaves) {
        List<ParameterSpace> unique = new ArrayList<>();
        for (ParameterSpace p : allLeaves) {
            //This isn't especially efficient, but small number of parameters in general means it's fine
            boolean found = false;
            for (ParameterSpace q : unique) {
                if (p == q) {
                    found = true;
                }
            }
            if (!found) {
                unique.add(p);
            }
        }

        return unique;
    }

    /**
     * Count the number of unique parameters in the specified leaf nodes
     *
     * @param allLeaves Leaf values to count the parameters fore
     * @return Number of parameters for all unique objects
     */
    public static int countUniqueParameters(List<ParameterSpace> allLeaves) {
        List<ParameterSpace> unique = getUniqueObjects(allLeaves);
        int count = 0;
        for (ParameterSpace ps : unique) {
            if (!ps.isLeaf()) {
                throw new IllegalStateException("Method should only be used with leaf nodes");
            }
            count += ps.numParameters();
        }
        return count;
    }

}
