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

package org.datavec.api.transform.analysis.quality.real;

import org.datavec.api.transform.quality.columns.DoubleQuality;
import org.nd4j.linalg.function.BiFunction;

import java.io.Serializable;

/**
 * Created by Alex on 5/03/2016.
 */
public class RealQualityMergeFunction implements BiFunction<DoubleQuality, DoubleQuality, DoubleQuality>, Serializable {
    @Override
    public DoubleQuality apply(DoubleQuality v1, DoubleQuality v2) {
        return v1.add(v2);
    }
}
