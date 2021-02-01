/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.python4j;

@lombok.Data
public class PythonVariable<T> {

    private String name;
    private String type;
    private T value;

    private static boolean validateVariableName(String s) {
        if (s.isEmpty()) return false;
        if (!Character.isJavaIdentifierStart(s.charAt(0))) return false;
        for (int i = 1; i < s.length(); i++)
            if (!Character.isJavaIdentifierPart(s.charAt(i)))
                return false;
        return true;
    }

    public PythonVariable(String name, PythonType<T> type, Object value) {
        if (!validateVariableName(name)) {
            throw new PythonException("Invalid identifier: " + name);
        }
        this.name = name;
        this.type = type.getName();
        setValue(value);
    }

    public PythonVariable(String name, PythonType<T> type) {
        this(name, type, null);
    }

    public PythonType<T> getType() {
        return PythonTypes.get(this.type);
    }

    public T getValue() {
        return this.value;
    }

    public void setValue(Object value) {
        this.value = value == null ? null : getType().adapt(value);
    }

    public PythonObject getPythonObject() {
        return getType().toPython(value);
    }

}
