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

package org.datavec.local.transforms.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.functions.FlatMapFunctionAdapter;

import java.util.Collections;
import java.util.List;

/**
 * Function for executing a transform process
 */
public class LocalTransformProcessFunctionAdapter implements FlatMapFunctionAdapter<List<Writable>, List<Writable>> {

    private final TransformProcess transformProcess;

    public LocalTransformProcessFunctionAdapter(TransformProcess transformProcess) {
        this.transformProcess = transformProcess;
    }

    @Override
    public List<List<Writable>> call(List<Writable> v1) throws Exception {
        List<Writable> newList = transformProcess.execute(v1);
        if (newList == null)
            return Collections.emptyList(); //Example was filtered out
        else
            return Collections.singletonList(newList);
    }
}
