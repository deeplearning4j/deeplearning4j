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
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/2/16.
 */
public class Basic extends VizDoom {

    public Basic(boolean render) {
        super(render);
    }

    public Configuration getConfiguration() {

        List<Button> buttons = Arrays.asList(Button.ATTACK, Button.MOVE_LEFT, Button.MOVE_RIGHT);

        return new Configuration("basic", -0.01, 1, 0, 700, 0, buttons);
    }

    public Basic newInstance() {
        return new Basic(isRender());
    }
}
