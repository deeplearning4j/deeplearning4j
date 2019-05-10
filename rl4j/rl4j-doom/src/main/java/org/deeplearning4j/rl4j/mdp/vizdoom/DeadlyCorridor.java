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

package org.deeplearning4j.rl4j.mdp.vizdoom;

import vizdoom.Button;

import java.util.Arrays;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/1/16.
 */
public class DeadlyCorridor extends VizDoom {

    public DeadlyCorridor(boolean render) {
        super(render);
    }

    public Configuration getConfiguration() {
        setScaleFactor(1.0);
        List<Button> buttons = Arrays.asList(Button.ATTACK, Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
                        Button.TURN_LEFT, Button.TURN_RIGHT);



        return new Configuration("deadly_corridor", 0.0, 5, 100, 2100, 0, buttons);
    }

    public DeadlyCorridor newInstance() {
        return new DeadlyCorridor(isRender());
    }
}

