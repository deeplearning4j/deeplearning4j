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
