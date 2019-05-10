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


import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import vizdoom.*;

import java.util.ArrayList;
import java.util.List;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/28/16.
 *
 * Mother abstract class for all VizDoom scenarios
 *
 * is mostly configured by
 *
 *    String scenario;       name of the scenario
 *    double livingReward;   additional reward at each step for living
 *    double deathPenalty;   negative reward when ded
 *    int doomSkill;         skill of the ennemy
 *    int timeout;           number of step after which simulation time out
 *    int startTime;         number of internal tics before the simulation starts (useful to draw weapon by example)
 *    List<Button> buttons;  the list of inputs one can press for a given scenario (noop is automatically added)
 *
 *
 *
 */
@Slf4j
abstract public class VizDoom implements MDP<VizDoom.GameScreen, Integer, DiscreteSpace> {


    final public static String DOOM_ROOT = "vizdoom";

    protected DoomGame game;
    final protected List<double[]> actions;
    final protected DiscreteSpace discreteSpace;
    final protected ObservationSpace<GameScreen> observationSpace;
    @Getter
    final protected boolean render;
    @Setter
    protected double scaleFactor = 1;

    public VizDoom() {
        this(false);
    }

    public VizDoom(boolean render) {
        this.render = render;
        actions = new ArrayList<double[]>();
        game = new DoomGame();
        setupGame();
        discreteSpace = new DiscreteSpace(getConfiguration().getButtons().size() + 1);
        observationSpace = new ArrayObservationSpace<>(new int[] {game.getScreenHeight(), game.getScreenWidth(), 3});
    }


    public void setupGame() {

        Configuration conf = getConfiguration();

        game.setViZDoomPath(DOOM_ROOT + "/vizdoom");
        game.setDoomGamePath(DOOM_ROOT + "/freedoom2.wad");
        game.setDoomScenarioPath(DOOM_ROOT + "/scenarios/" + conf.getScenario() + ".wad");

        game.setDoomMap("map01");

        game.setScreenFormat(ScreenFormat.RGB24);
        game.setScreenResolution(ScreenResolution.RES_800X600);
        // Sets other rendering options
        game.setRenderHud(false);
        game.setRenderCrosshair(false);
        game.setRenderWeapon(true);
        game.setRenderDecals(false);
        game.setRenderParticles(false);


        GameVariable[] gameVar = new GameVariable[] {GameVariable.KILLCOUNT, GameVariable.ITEMCOUNT,
                        GameVariable.SECRETCOUNT, GameVariable.FRAGCOUNT, GameVariable.HEALTH, GameVariable.ARMOR,
                        GameVariable.DEAD, GameVariable.ON_GROUND, GameVariable.ATTACK_READY,
                        GameVariable.ALTATTACK_READY, GameVariable.SELECTED_WEAPON, GameVariable.SELECTED_WEAPON_AMMO,
                        GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4,
                        GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8,
                        GameVariable.AMMO9, GameVariable.AMMO0};
        // Adds game variables that will be included in state.

        for (int i = 0; i < gameVar.length; i++) {
            game.addAvailableGameVariable(gameVar[i]);
        }


        // Causes episodes to finish after timeout tics
        game.setEpisodeTimeout(conf.getTimeout());

        game.setEpisodeStartTime(conf.getStartTime());

        game.setWindowVisible(render);
        game.setSoundEnabled(false);
        game.setMode(Mode.PLAYER);


        game.setLivingReward(conf.getLivingReward());

        // Adds buttons that will be allowed.
        List<Button> buttons = conf.getButtons();
        int size = buttons.size();

        actions.add(new double[size + 1]);
        for (int i = 0; i < size; i++) {
            game.addAvailableButton(buttons.get(i));
            double[] action = new double[size + 1];
            action[i] = 1;
            actions.add(action);
        }

        game.setDeathPenalty(conf.getDeathPenalty());
        game.setDoomSkill(conf.getDoomSkill());

        game.init();
    }

    public boolean isDone() {
        return game.isEpisodeFinished();
    }


    public GameScreen reset() {
        log.info("free Memory: " + Pointer.formatBytes(Pointer.availablePhysicalBytes()) + "/"
                        + Pointer.formatBytes(Pointer.totalPhysicalBytes()));

        game.newEpisode();
        return new GameScreen(game.getState().screenBuffer);
    }


    public void close() {
        game.close();
    }


    public StepReply<GameScreen> step(Integer action) {

        double r = game.makeAction(actions.get(action)) * scaleFactor;
        log.info(game.getEpisodeTime() + " " + r + " " + action + " ");
        return new StepReply(new GameScreen(game.isEpisodeFinished()
                ? new byte[game.getScreenSize()]
                : game.getState().screenBuffer), r, game.isEpisodeFinished(), null);

    }


    public ObservationSpace<GameScreen> getObservationSpace() {
        return observationSpace;
    }


    public DiscreteSpace getActionSpace() {
        return discreteSpace;
    }

    public abstract Configuration getConfiguration();

    public abstract VizDoom newInstance();

    @Value
    public static class Configuration {
        String scenario;
        double livingReward;
        double deathPenalty;
        int doomSkill;
        int timeout;
        int startTime;
        List<Button> buttons;
    }

    public static class GameScreen implements Encodable {


        double[] array;

        public GameScreen(byte[] screen) {
            array = new double[screen.length];
            for (int i = 0; i < screen.length; i++) {
                array[i] = (screen[i] & 0xFF) / 255.0;
            }
        }

        public double[] toArray() {
            return array;
        }
    }

}
