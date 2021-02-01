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

package org.datavec.local.transforms.sequence;

import lombok.AllArgsConstructor;
import org.datavec.api.writable.Writable;
import org.nd4j.common.function.Function;
import org.nd4j.common.primitives.Pair;

import java.util.ArrayList;
import java.util.List;

@AllArgsConstructor
public class LocalMapToPairByMultipleColumnsFunction
                implements Function<List<Writable>, Pair<List<Writable>, List<Writable>>> {

    private final int[] keyColumnIdxs;

    @Override
    public Pair<List<Writable>, List<Writable>> apply(List<Writable> writables) {
        List<Writable> keyOut = new ArrayList<>(keyColumnIdxs.length);
        for (int keyColumnIdx : keyColumnIdxs) {
            keyOut.add(writables.get(keyColumnIdx));
        }
        return Pair.of(keyOut, writables);
    }
}
