package org.deeplearning4j.rl4j.mdp.vizdoom;

import vizdoom.Button;

import java.util.Arrays;
import java.util.List;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/18/16.
 */
public class PredictPosition extends VizDoom {

    public PredictPosition(boolean render) {
        super(render);
    }

    public Configuration getConfiguration() {
        setScaleFactor(1.0);
        List<Button> buttons = Arrays.asList(Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK);
        return new Configuration("predict_position", -0.0001, 1, 0, 2100, 35, buttons);
    }

    public PredictPosition newInstance() {
        return new PredictPosition(isRender());
    }
}


