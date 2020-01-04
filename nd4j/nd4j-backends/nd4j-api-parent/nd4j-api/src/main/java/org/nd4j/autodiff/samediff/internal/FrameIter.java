/* ******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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

package org.nd4j.autodiff.samediff.internal;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * FrameIter: Identifies a frame + iteration (but not a specific op or variable).<br>
 * Note that frames can be nested - which generally represents nested loop situations.
 */
@Data
@AllArgsConstructor
public class FrameIter {
    private String frame;
    private int iteration;
    private FrameIter parentFrame;

    @Override
    public String toString() {
        return "(\"" + frame + "\"," + iteration + (parentFrame == null ? "" : ",parent=" + parentFrame.toString()) + ")";
    }

    @Override
    public FrameIter clone() {
        return new FrameIter(frame, iteration, (parentFrame == null ? null : parentFrame.clone()));
    }

    public AbstractSession.VarId toVarId(String name) {
        return new AbstractSession.VarId(name, frame, iteration, parentFrame);
    }
}
