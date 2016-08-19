package org.deeplearning4j.rl4j.mdp.vizdoom;


import lombok.Getter;
import lombok.Setter;
import lombok.Value;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.rl4j.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import oshi.SystemInfo;
import oshi.hardware.GlobalMemory;
import oshi.util.FormatUtil;
import vizdoom.*;

import java.util.ArrayList;
import java.util.List;


/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/28/16.
 */
abstract public class VizDoom implements MDP<VizDoom.GameScreen, Integer, DiscreteSpace> {


    final public static String DOOM_ROOT = "vizdoom";
    
    protected DoomGame game;
    final protected Logger log = LoggerFactory.getLogger("Vizdoom");
    final protected GlobalMemory memory = new SystemInfo().getHardware().getMemory();
    final protected List<int[]> actions;
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
        actions = new ArrayList<int[]>();
        game = new DoomGame();
        setupGame();
        discreteSpace = new DiscreteSpace(getConfiguration().getButtons().size() + 1);
        observationSpace = new ArrayObservationSpace<GameScreen>(new int[]{game.getScreenHeight(), game.getScreenWidth(), 3});
    }


    public void setupGame() {

        Configuration conf = getConfiguration();

        game.setViZDoomPath(DOOM_ROOT + "/vizdoom");
        game.setDoomGamePath(DOOM_ROOT + "/scenarios/freedoom2.wad");
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


        GameVariable[] gameVar = new GameVariable[]{
                GameVariable.KILLCOUNT,
                GameVariable.ITEMCOUNT,
                GameVariable.SECRETCOUNT,
                GameVariable.FRAGCOUNT,
                GameVariable.HEALTH,
                GameVariable.ARMOR,
                GameVariable.DEAD,
                GameVariable.ON_GROUND,
                GameVariable.ATTACK_READY,
                GameVariable.ALTATTACK_READY,
                GameVariable.SELECTED_WEAPON,
                GameVariable.SELECTED_WEAPON_AMMO,
                GameVariable.AMMO1,
                GameVariable.AMMO2,
                GameVariable.AMMO3,
                GameVariable.AMMO4,
                GameVariable.AMMO5,
                GameVariable.AMMO6,
                GameVariable.AMMO7,
                GameVariable.AMMO8,
                GameVariable.AMMO9,
                GameVariable.AMMO0
        };
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

        actions.add(new int[size + 1]);
        for (int i = 0; i < size; i++) {
            game.addAvailableButton(buttons.get(i));
            int[] action = new int[size + 1];
            action[i] = 1;
            actions.add(action);
        }


        game.init();
    }

    public boolean isDone() {
        return game.isEpisodeFinished();
    }


    public GameScreen reset() {
        log.info("free Memory: " + FormatUtil.formatBytes(memory.getAvailable()) + "/"
                + FormatUtil.formatBytes(memory.getTotal()));

        game.newEpisode();
        game.getGameScreen();
        return new GameScreen(game.getGameScreen());
    }


    public void close() {
        game.close();
    }


    public StepReply<GameScreen> step(Integer action) {

        double r = 0;
        log.info("action: " + action + " episode:" + game.getEpisodeTime() + " javacpp:" + FormatUtil.formatBytes(Pointer.totalBytes()));
        //try {
            r = game.makeAction(actions.get(action)) * scaleFactor;
            /*
        } catch (ViZDoomErrorException e) {
            r = -2000;
            game = new DoomGame();
            setupGame();
            return new StepReply<>(new GameScreen(game.getGameScreen()), r, true, null);
        }
        //System.out.println(r + " " + scaleFactor);
        */
        return new StepReply(new GameScreen(game.getGameScreen()), r, game.isEpisodeFinished(), null);
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
        int timeout;
        int startTime;
        List<Button> buttons;
    }

    public static class GameScreen implements Encodable {


        double[] array;

        public GameScreen(int[] screen) {
            array = new double[screen.length];
            for (int i = 0; i < screen.length; i++) {
                array[i] = screen[i];
            }
        }

        public double[] toArray() {
            return array;
        }
    }

}
