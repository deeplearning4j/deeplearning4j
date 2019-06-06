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

package vizdoom;

public class GameState{
    public int number;
    public int tic;

    public double[] gameVariables;

    public byte[] screenBuffer;
    public byte[] depthBuffer;
    public byte[] labelsBuffer;
    public byte[] automapBuffer;

    public Label[] labels;

    GameState(int number,
        int tic,
        double[] gameVariables,
        byte[] screenBuffer,
        byte[] depthBuffer,
        byte[] labelsBuffer,
        byte[] automapBuffer,
        Label[] labels){

        this.number = number;
        this.tic = tic;
        this.gameVariables = gameVariables;
        this.screenBuffer = screenBuffer;
        this.depthBuffer = depthBuffer;
        this.labelsBuffer = labelsBuffer;
        this.automapBuffer = automapBuffer;
        this.labels = labels;
    }
}
