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

package org.nd4j.autodiff.samediff;

import java.util.*;

/**
 * Detailed frame metadata for execution analysis
 */
public class FrameMetadata {
    public String frameName;
    public String parentFrame;
    public Set<String> childFrames = new HashSet<>();
    public int depth;
    public FrameType frameType;
    public int totalOperations;
    public int totalVariables;
    public boolean hasLoops;
    public boolean hasConditionals;
    public int maxIterations;
    public List<String> entryPoints = new ArrayList<>();
    public List<String> exitPoints = new ArrayList<>();
    public Map<FrameTransition, Integer> transitionCounts = new HashMap<>();
    
    public FrameMetadata(String frameName, String parentFrame, int depth, FrameType type) {
        this.frameName = frameName;
        this.parentFrame = parentFrame;
        this.depth = depth;
        this.frameType = type;
    }
}