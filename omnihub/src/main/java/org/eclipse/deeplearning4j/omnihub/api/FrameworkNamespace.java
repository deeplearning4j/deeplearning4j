/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.omnihub.api;

import org.eclipse.deeplearning4j.omnihub.Framework;

import java.util.Locale;

public enum FrameworkNamespace {
    DL4J,SAMEDIFF;

    public static FrameworkNamespace fromString(String namespace) {
        switch(namespace.toLowerCase()) {
            case "dl4j":
                return DL4J;
            case "samediff":
                return SAMEDIFF;
            default:
                return null;
        }
    }

    public  String javaClassName() {
        switch(this) {
            case DL4J:
                return "DL4J";
            case SAMEDIFF:
                return "SAMEDIFF";
        }

        throw new IllegalStateException("Unable to determine java class type. Invalid namespace type " + this.name());
    }

}
