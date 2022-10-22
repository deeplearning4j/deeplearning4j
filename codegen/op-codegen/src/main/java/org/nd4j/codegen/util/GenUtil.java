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

package org.nd4j.codegen.util;

import org.nd4j.codegen.api.Op;

public class GenUtil {

    private GenUtil(){ }

    public static String ensureFirstIsCap(String in){
        if(Character.isUpperCase(in.charAt(0))){
           return in;
        }

        return Character.toUpperCase(in.charAt(0)) + in.substring(1);
    }

    public static String ensureFirstIsNotCap(String in){
        if(Character.isLowerCase(in.charAt(0))){
            return in;
        }

        return Character.toLowerCase(in.charAt(0)) + in.substring(1);
    }

    public static String repeat(String in, int count){
        StringBuilder sb = new StringBuilder();
        for( int i=0; i<count; i++ ){
            sb.append(in);
        }
        return sb.toString();
    }

    public static String addIndent(String in, int count){
        if(in == null)
            return null;
        String[] lines = in.split("\n");
        StringBuilder out = new StringBuilder();
        String indent = repeat(" ", count);
        for(String s : lines){
            out.append(indent).append(s).append("\n");
        }
        return out.toString();
    }
}
