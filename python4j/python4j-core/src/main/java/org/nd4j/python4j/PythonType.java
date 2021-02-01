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


import java.io.File;

public abstract class PythonType<T> {

    private final String name;
    private final Class<T> javaType;

    public PythonType(String name, Class<T> javaType) {
        this.name = name;
        this.javaType = javaType;
    }

    public T adapt(Object javaObject) throws PythonException {
        return (T) javaObject;
    }

    public abstract T toJava(PythonObject pythonObject);

    public abstract PythonObject toPython(T javaObject);

    public boolean accepts(Object javaObject) {
        return javaType.isAssignableFrom(javaObject.getClass());
    }

    public String getName() {
        return name;
    }

    @Override
    public boolean equals(Object obj){
        if (!(obj instanceof PythonType)){
            return false;
        }
        PythonType other = (PythonType)obj;
        return this.getClass().equals(other.getClass()) && this.name.equals(other.name);
    }

    public PythonObject pythonType(){
        return null;
    }

    public File[] packages(){
        return new File[0];
    }

    public void init(){ //not to be called from constructor

    }

}
