package org.deeplearning4j.rl4j.mdp.vizdoom;

import vizdoom.Button;

import java.util.Arrays;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/1/16.
 */
public class TakeCover extends VizDoom {

    public TakeCover(boolean render) {
        super(render);
    }

    public Configuration getConfiguration() {
        setScaleFactor(1.0);
        List<Button> buttons = Arrays.asList(Button.MOVE_LEFT, Button.MOVE_RIGHT);
        return new Configuration("take_cover", 1, 1, 0, 2100, 0, buttons);
    }

    public TakeCover newInstance() {
        return new TakeCover(isRender());
    }
}

