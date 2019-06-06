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

package org.deeplearning4j.arbiter.optimize.runner;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Simple helper class to store status of a candidate that is/has been/will be executed
 */
@AllArgsConstructor
@Data
public class CandidateInfo {

    public CandidateInfo() {
        //No arg constructor for Jackson
    }

    private int index;
    private CandidateStatus candidateStatus;
    private Double score;
    private long createdTime;
    private Long startTime;
    private Long endTime;
    private double[] flatParams; //Same as parameters in Candidate class
    private String exceptionStackTrace;
}
