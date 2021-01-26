/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.mdp.ale;

import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.ale.ALEInterface;
import org.bytedeco.javacpp.IntPointer;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author saudet
 */
@Slf4j
public class ALEMDP implements MDP<ALEMDP.GameScreen, Integer, DiscreteSpace> {
    protected ALEInterface ale;
    final protected int[] actions;
    final protected DiscreteSpace discreteSpace;
    final protected ObservationSpace<GameScreen> observationSpace;
    @Getter
    final protected String romFile;
    @Getter
    final protected boolean render;
    @Getter
    final protected Configuration configuration;
    @Setter
    protected double scaleFactor = 1;

    private byte[] screenBuffer;

    public ALEMDP(String romFile) {
        this(romFile, false);
    }

    public ALEMDP(String romFile, boolean render) {
        this(romFile, render, new Configuration(123, 0, 0, 0, true));
    }

    public ALEMDP(String romFile, boolean render, Configuration configuration) {
        this.romFile = romFile;
        this.configuration = configuration;
        this.render = render;
        ale = new ALEInterface();
        setupGame();

        // Get the vector of minimal or legal actions
        IntPointer a = (getConfiguration().minimalActionSet ? ale.getMinimalActionSet()
                        : ale.getLegalActionSet());
        actions = new int[(int)a.limit()];
        a.get(actions);

        int height = (int)ale.getScreen().height();
        int width = (int)(int)ale.getScreen().width();

        discreteSpace = new DiscreteSpace(actions.length);
        int[] shape = {3, height, width};
        observationSpace = new ArrayObservationSpace<>(shape);
        screenBuffer = new byte[shape[0] * shape[1] * shape[2]];

    }

    public void setupGame() {
        Configuration conf = getConfiguration();

        // Get & Set the desired settings
        ale.setInt("random_seed", conf.randomSeed);
        ale.setFloat("repeat_action_probability", conf.repeatActionProbability);

        ale.setBool("display_screen", render);
        ale.setBool("sound", render);

        // Causes episodes to finish after timeout tics
        ale.setInt("max_num_frames", conf.maxNumFrames);
        ale.setInt("max_num_frames_per_episode", conf.maxNumFramesPerEpisode);

        // Load the ROM file. (Also resets the system for new settings to
        // take effect.)
        ale.loadROM(romFile);
    }

    public boolean isDone() {
        return ale.game_over();
    }


    public GameScreen reset() {
        ale.reset_game();
        ale.getScreenRGB(screenBuffer);
        return new GameScreen(observationSpace.getShape(), screenBuffer);
    }


    public void close() {
        ale.deallocate();
    }

    public StepReply<GameScreen> step(Integer action) {
        double r = ale.act(actions[action]) * scaleFactor;
        log.info(ale.getEpisodeFrameNumber() + " " + r + " " + action + " ");
        ale.getScreenRGB(screenBuffer);

        return new StepReply(new GameScreen(observationSpace.getShape(), screenBuffer), r, ale.game_over(), null);
    }

    public ObservationSpace<GameScreen> getObservationSpace() {
        return observationSpace;
    }

    public DiscreteSpace getActionSpace() {
        return discreteSpace;
    }

    public ALEMDP newInstance() {
        return new ALEMDP(romFile, render, configuration);
    }

    @Value
    public static class Configuration {
        int randomSeed;
        float repeatActionProbability;
        int maxNumFrames;
        int maxNumFramesPerEpisode;
        boolean minimalActionSet;
    }

    public static class GameScreen implements Encodable {

        final INDArray data;
        public GameScreen(int[] shape, byte[] screen) {

            data = Nd4j.create(screen, new long[] {shape[1], shape[2], 3}, DataType.UINT8).permute(2,0,1);
        }

        private GameScreen(INDArray toDup) {
            data = toDup.dup();
        }

        @Override
        public double[] toArray() {
            return data.data().asDouble();
        }

        @Override
        public boolean isSkipped() {
            return false;
        }

        @Override
        public INDArray getData() {
            return data;
        }

        @Override
        public GameScreen dup() {
            return new GameScreen(data);
        }
    }
}
