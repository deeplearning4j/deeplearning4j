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

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.enums;

/**
 * Effective only for the ResizeNearest interpolation.
 * Indicates how to get "nearest" pixel in NDArray from original coordinate
 * FLOOR = the largest integer value not greater than
 * ROUND_PREFER_FLOOR = round half down
 * ROUND_PREFER_CEIL = round half up
 * CEIL =  nearest integer not less than
 */
public enum NearestMode {
    FLOOR(0),
    ROUND_PREFER_FLOOR(1),
    ROUND_PREFER_CEIL(2),
    CEIL(3);

    private final int index;
    NearestMode(int index) { this.index = index; }
    public int getIndex() { return index; }
}
