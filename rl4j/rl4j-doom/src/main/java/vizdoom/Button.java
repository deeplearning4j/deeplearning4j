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
public enum Button{
    ATTACK,
    USE,
    JUMP,
    CROUCH,
    TURN180,
    ALTATTACK,
    RELOAD,
    ZOOM,

    SPEED,
    STRAFE,

    MOVE_RIGHT,
    MOVE_LEFT,
    MOVE_BACKWARD,
    MOVE_FORWARD,
    TURN_RIGHT,
    TURN_LEFT,
    LOOK_UP,
    LOOK_DOWN,
    MOVE_UP,
    MOVE_DOWN,
    LAND,

    SELECT_WEAPON1,
    SELECT_WEAPON2,
    SELECT_WEAPON3,
    SELECT_WEAPON4,
    SELECT_WEAPON5,
    SELECT_WEAPON6,
    SELECT_WEAPON7,
    SELECT_WEAPON8,
    SELECT_WEAPON9,
    SELECT_WEAPON0,

    SELECT_NEXT_WEAPON,
    SELECT_PREV_WEAPON,
    DROP_SELECTED_WEAPON,

    ACTIVATE_SELECTED_ITEM,
    SELECT_NEXT_ITEM,
    SELECT_PREV_ITEM,
    DROP_SELECTED_ITEM,

    LOOK_UP_DOWN_DELTA,
    TURN_LEFT_RIGHT_DELTA,
    MOVE_FORWARD_BACKWARD_DELTA,
    MOVE_LEFT_RIGHT_DELTA,
    MOVE_UP_DOWN_DELTA;
};
