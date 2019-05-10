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

/// <reference path="../api/Margin.ts" />

class TSUtils {

    //Get the maximum value
    static max(input: number[][]): number {
        var max: number = -Number.MAX_VALUE;
        for (var i = 0; i < input.length; i++) {
            for( var j=0; j<input[i].length; j++ ){
                max = Math.max(max,input[i][j]);
            }
        }
        return max;
    }

    //Get the minimum value
    static min(input: number[][]): number {
        var min: number = Number.MAX_VALUE;
        for (var i = 0; i < input.length; i++) {
            for( var j=0; j<input[i].length; j++ ){
                min = Math.min(min,input[i][j]);
            }
        }
        return min;
    }

    //Normalize the length unit (for example, parse the LengthUnit enum values)
    static normalizeLengthUnit(input: string): string{
        if(input == null) return input;

        switch(input.toLowerCase()){
            case "px":
                return "px";
            case "percent":
            case "%":
                return "%";
            case "cm":
                return "cm";
            case "mm":
                return "mm";
            case "in":
                return "in";
            default:
                return input;
        }

    }
}