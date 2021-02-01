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

package org.datavec.api.transform.analysis.quality.longq;

import org.datavec.api.transform.quality.columns.LongQuality;
import org.nd4j.common.function.BiFunction;

import java.io.Serializable;

public class LongQualityMergeFunction implements BiFunction<LongQuality, LongQuality, LongQuality>, Serializable {
    @Override
    public LongQuality apply(LongQuality v1, LongQuality v2) {
        return v1.add(v2);
    }
}
