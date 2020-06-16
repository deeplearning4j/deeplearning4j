/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.eclipse.python4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Some syntax sugar for lookup by name
 */
public class PythonVariables extends ArrayList<PythonVariable> {
    public PythonVariable get(String variableName) {
        for (PythonVariable pyVar: this){
            if (pyVar.getName().equals(variableName)){
                return pyVar;
            }
        }
        return null;
    }

    public <T> boolean add(String variableName, PythonType<T> variableType, Object value){
        return this.add(new PythonVariable<>(variableName, variableType, value));
    }

    public PythonVariables(PythonVariable... variables){
        this(Arrays.asList(variables));
    }
    public PythonVariables(List<PythonVariable> list){
        super();
        addAll(list);
    }
}
