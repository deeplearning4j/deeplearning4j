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
package org.eclipse.deeplearning4j.omnihub;

/**
 * Represents a framework for model conversion and manipulation
 *
 * @author Adam Gibson
 */
public enum Framework {
    SAMEDIFF(0),
    PYTORCH(1),
    TENSORFLOW(2),
    KERAS(3),
    DL4J(4),
    ONNX(5),
    HUGGINGFACE(6);

    private final int frameworkIndex;
    Framework(int index) { this.frameworkIndex = index; }
    public int frameworkIndex() { return frameworkIndex; }

    /**
     * Returns true if the framework is an input framework (pytorch, keras, tensorflow,onnx)
     * @param framework the input framework
     * @return
     */
    public static boolean isInput(Framework framework) {
        switch(framework) {
            case TENSORFLOW:
            case KERAS:
            case PYTORCH:
            case ONNX:
                return true;
            default:
                return false;
        }
    }

    /**
     * Returns true if the framework is an output framework (dl4j or samediff)
     * @param framework the input framework
     * @return
     */
    public static boolean isOutput(Framework framework) {
        return !isInput(framework);
    }

    /**
     * Return the output framework for a given framework.
     * Most of the time it will be samediff, but keras's h5 format will use dl4j.
     * Note an {@link IllegalArgumentException} will be thrown for either {@link #SAMEDIFF}
     * or {@link #DL4J}
     * @param framework the input framework
     * @return the appropriate output framework for the given input framework.
     */
    public static Framework outputFrameworkFor(Framework framework) {
        if(!isInput(framework)) {
            throw new IllegalArgumentException("Input framework " + framework.name() + " is not an input framework");
        }

        switch(framework) {
            case ONNX:
            case PYTORCH:
            case TENSORFLOW:
                return SAMEDIFF;
            default:
                return DL4J;
        }
    }

}
