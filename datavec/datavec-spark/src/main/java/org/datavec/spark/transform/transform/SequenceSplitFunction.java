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

package org.datavec.spark.transform.transform;

import org.datavec.api.transform.sequence.SequenceSplit;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.BaseFlatMapFunctionAdaptee;

import java.util.List;

/**
 * Created by Alex on 17/03/2016.
 */
public class SequenceSplitFunction extends BaseFlatMapFunctionAdaptee<List<List<Writable>>, List<List<Writable>>> {

    public SequenceSplitFunction(SequenceSplit split) {
        super(new SequenceSplitFunctionAdapter(split));
    }

}
