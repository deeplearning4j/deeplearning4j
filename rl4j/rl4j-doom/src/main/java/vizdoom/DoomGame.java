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

public class DoomGame{
    static {
        System.loadLibrary("vizdoom");
    }

    public long internalPtr = 0;
    public DoomGame(){
        this.DoomGameNative();
    }

    public native double doomTics2Ms(double tics, int ticrate);
    public native double ms2DoomTics(double ms, int ticrate);
    public native double doomTics2Sec(double tics, int ticrate);
    public native double sec2DoomTics(double sec, int ticrate);
    public native double doomFixedToDouble(double doomFixed);
    public native boolean isBinaryButton(Button button);
    public native boolean isDeltaButton(Button button);

    private native void DoomGameNative();
    public native boolean loadConfig(String file);

    public native boolean init();
    public native void close();

    public native void newEpisode();
    public native void newEpisode(String file);
    public native void replayEpisode(String file);
    public native void replayEpisode(String file, int player);

    public native boolean isRunning();
    public native boolean isMultiplayerGame();

    public native void setAction(double[] actions);
    public native void advanceAction();
    public native void advanceAction(int tics);
    public native void advanceAction(int tics, boolean stateUpdate);
    public native double makeAction(double[] actions);
    public native double makeAction(double[] actions, int tics);

    public native GameState getState();

    public native double[] getLastAction();

    public native boolean isNewEpisode();
    public native boolean isEpisodeFinished();

    public native boolean isPlayerDead();
    public native void respawnPlayer();

    public native Button[] getAvailableButtons();
    public native void setAvailableButtons(Button[] buttons);
    public native void addAvailableButton(Button button);
    public native void addAvailableButton(Button button, double maxValue);
    public native void clearAvailableButtons();
    public native int getAvailableButtonsSize();

    public native void setButtonMaxValue(Button button, double maxValue);
    public native double getButtonMaxValue(Button button);

    public native GameVariable[] getAvailableGameVariables();
    public native void setAvailableGameVariables(GameVariable[] gameVariables);
    public native void addAvailableGameVariable(GameVariable var);
    public native void clearAvailableGameVariables();
    public native int getAvailableGameVariablesSize();

    public native void addGameArgs(String arg);
    public native void clearGameArgs();

    public native void sendGameCommand(String cmd);

    private native int getModeNative();

    public Mode getMode(){
        return Mode.values()[this.getModeNative()];
    }

    public native void setMode(Mode mode);

    public native int getTicrate();
    public native void setTicrate(int ticrate);

    public native double getGameVariable(GameVariable var);

    public native double getLivingReward();
    public native void setLivingReward(double livingReward);
    public native double getDeathPenalty();
    public native void setDeathPenalty(double deathPenalty);

    public native double getLastReward();
    public native double getTotalReward();

    public native void setViZDoomPath(String path);
    public native void setDoomGamePath(String path);
    public native void setDoomScenarioPath(String path);
    public native void setDoomMap(String map);
    public native void setDoomSkill(int skill);
    public native void setDoomConfigPath(String path);

    public native int getSeed();
    public native void setSeed(int seed);

    public native int getEpisodeStartTime();
    public native void setEpisodeStartTime(int tics);

    public native int getEpisodeTimeout();
    public native void setEpisodeTimeout(int tics);

    public native int getEpisodeTime();

    public native boolean isDepthBufferEnabled();
    public native void setDepthBufferEnabled(boolean depthBuffer);

    public native boolean isLabelsBufferEnabled();
    public native void setLabelsBufferEnabled(boolean labelsBuffer);

    public native boolean isAutomapBufferEnabled();
    public native void setAutomapBufferEnabled(boolean automapBuffer);
    public native void setAutomapMode(AutomapMode mode);
    public native void setAutomapRotate(boolean rotate);
    public native void setAutomapRenderTextures(boolean textures);

    public native void setScreenResolution(ScreenResolution resolution);
    public native void setScreenFormat(ScreenFormat format);
    public native void setRenderHud(boolean hud);
    public native void setRenderMinimalHud(boolean minimalHud);
    public native void setRenderWeapon(boolean weapon);
    public native void setRenderCrosshair(boolean crosshair);
    public native void setRenderDecals(boolean decals);
    public native void setRenderParticles(boolean particles);
    public native void setRenderEffectsSprites(boolean sprites);
    public native void setRenderMessages(boolean messages);
    public native void setRenderCorpses(boolean corpses);
    public native void setRenderScreenFlashes(boolean flashes);
    public native void setRenderAllFrames(boolean allFrames);
    public native void setWindowVisible(boolean visibility);
    public native void setConsoleEnabled(boolean console);
    public native void setSoundEnabled(boolean sound);

    public native int getScreenWidth();
    public native int getScreenHeight();
    public native int getScreenChannels();
    public native int getScreenPitch();
    public native int getScreenSize();
    private native int getScreenFormatNative();

    public ScreenFormat getScreenFormat(){
        return ScreenFormat.values()[this.getScreenFormatNative()];
    }

}
