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

package org.nd4j.autodiff.samediff.execution;

import lombok.Data;

/**
 * Frame context information for control flow operations.
 * In TensorFlow's execution model, operations can execute in different "frames"
 * which represent different scopes (like loop bodies). Each frame can have
 * multiple iterations for loops.
 * 
 * @author Alex Gibson
 */
@Data
public class FrameInfo {
    
    private final String frameName;
    private final int iteration;
    private final FrameInfo parentFrame;
    
    /** The main/outer frame - default execution context */
    public static final FrameInfo OUTER_FRAME = new FrameInfo("main", 0, null);
    
    /**
     * Create a new frame info
     * 
     * @param frameName Name of the frame (e.g., "while_loop_1", "main")
     * @param iteration Current iteration number (0-based)
     * @param parentFrame Parent frame context, null for top-level
     */
    public FrameInfo(String frameName, int iteration, FrameInfo parentFrame) {
        this.frameName = frameName;
        this.iteration = iteration;
        this.parentFrame = parentFrame;
    }
    
    /**
     * Create a child frame with the same iteration
     */
    public FrameInfo createChildFrame(String childFrameName) {
        return new FrameInfo(childFrameName, 0, this);
    }
    
    /**
     * Create a new frame info with incremented iteration
     */
    public FrameInfo nextIteration() {
        return new FrameInfo(frameName, iteration + 1, parentFrame);
    }
    
    /**
     * Check if this frame is a child of the given frame
     */
    public boolean isChildOf(FrameInfo other) {
        FrameInfo current = this.parentFrame;
        while (current != null) {
            if (current.equals(other)) {
                return true;
            }
            current = current.parentFrame;
        }
        return false;
    }
    
    /**
     * Get the depth of this frame (0 for outer frame)
     */
    public int getDepth() {
        int depth = 0;
        FrameInfo current = this.parentFrame;
        while (current != null) {
            depth++;
            current = current.parentFrame;
        }
        return depth;
    }
    
    /**
     * Get a human-readable frame path
     */
    public String getFramePath() {
        if (parentFrame == null) {
            return frameName + ":" + iteration;
        } else {
            return parentFrame.getFramePath() + "/" + frameName + ":" + iteration;
        }
    }
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        
        FrameInfo frameInfo = (FrameInfo) o;
        
        if (iteration != frameInfo.iteration) return false;
        if (!frameName.equals(frameInfo.frameName)) return false;
        return parentFrame != null ? parentFrame.equals(frameInfo.parentFrame) : frameInfo.parentFrame == null;
    }
    
    @Override
    public int hashCode() {
        int result = frameName.hashCode();
        result = 31 * result + iteration;
        result = 31 * result + (parentFrame != null ? parentFrame.hashCode() : 0);
        return result;
    }
    
    @Override
    public String toString() {
        return getFramePath();
    }
}
