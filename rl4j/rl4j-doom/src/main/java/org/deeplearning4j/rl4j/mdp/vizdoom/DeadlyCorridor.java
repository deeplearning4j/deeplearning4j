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

