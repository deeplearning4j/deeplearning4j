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
package org.nd4j.autodiff.listeners.profiler.comparison;

import lombok.Builder;
import lombok.Data;
import lombok.experimental.Accessors;
import org.nd4j.common.function.BiFunction;

import java.io.File;

@Data
@Accessors(fluent = true)
@Builder
public class Config {

    private String p1Name;
    private String p2Name;
    private File profile1;
    private File profile2;
    private boolean profile1IsDir;
    private boolean profile2IsDir;
    @Builder.Default private ProfileAnalyzer.ProfileFormat profile1Format = ProfileAnalyzer.ProfileFormat.SAMEDIFF;
    @Builder.Default private ProfileAnalyzer.ProfileFormat profile2Format = ProfileAnalyzer.ProfileFormat.SAMEDIFF;
    @Builder.Default private ProfileAnalyzer.SortBy sortBy = ProfileAnalyzer.SortBy.PROFILE1_PC;
    private BiFunction<OpStats,OpStats,Boolean> filter;     //Return true to keep, false to remove
    @Builder.Default private ProfileAnalyzer.OutputFormat format = ProfileAnalyzer.OutputFormat.TEXT;

}
