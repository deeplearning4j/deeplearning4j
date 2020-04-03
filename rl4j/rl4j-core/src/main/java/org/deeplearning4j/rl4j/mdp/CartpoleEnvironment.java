package org.deeplearning4j.rl4j.mdp;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.environment.ActionSchema;
import org.deeplearning4j.rl4j.environment.Environment;
import org.deeplearning4j.rl4j.environment.Schema;
import org.deeplearning4j.rl4j.environment.StepResult;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class CartpoleEnvironment implements Environment<Integer> {
    private static final int NUM_ACTIONS = 2;
    private static final int ACTION_LEFT = 0;
    private static final int ACTION_RIGHT = 1;

    private static final Schema<Integer> schema = new Schema<>(new ActionSchema<>(ACTION_LEFT));

    public enum KinematicsIntegrators { Euler, SemiImplicitEuler };

    private static final double gravity = 9.8;
    private static final double massCart = 1.0;
    private static final double massPole = 0.1;
    private static final double totalMass = massPole + massCart;
    private static final double length = 0.5; // actually half the pole's length
    private static final double polemassLength = massPole * length;
    private static final double forceMag = 10.0;
    private static final double tau = 0.02;  // seconds between state updates

    // Angle at which to fail the episode
    private static final double thetaThresholdRadians = 12.0 * 2.0 * Math.PI / 360.0;
    private static final double xThreshold = 2.4;

    private final Random rnd;

    @Getter @Setter
    private KinematicsIntegrators kinematicsIntegrator = KinematicsIntegrators.Euler;

    @Getter
    private boolean episodeFinished = false;

    private double x;
    private double xDot;
    private double theta;
    private double thetaDot;
    private Integer stepsBeyondDone;

    public CartpoleEnvironment() {
        rnd = new Random();
    }

    public CartpoleEnvironment(int seed) {
        rnd = new Random(seed);
    }

    @Override
    public Schema<Integer> getSchema() {
        return schema;
    }

    @Override
    public Map<String, Object> reset() {

        x = 0.1 * rnd.nextDouble() - 0.05;
        xDot = 0.1 * rnd.nextDouble() - 0.05;
        theta = 0.1 * rnd.nextDouble() - 0.05;
        thetaDot = 0.1 * rnd.nextDouble() - 0.05;
        stepsBeyondDone = null;
        episodeFinished = false;

        return new HashMap<>() {{
            put("data", new double[]{x, xDot, theta, thetaDot});
        }};
    }

    @Override
    public StepResult step(Integer action) {
        double force = action == ACTION_RIGHT ? forceMag : -forceMag;
        double cosTheta = Math.cos(theta);
        double sinTheta = Math.sin(theta);
        double temp = (force + polemassLength * thetaDot * thetaDot * sinTheta) / totalMass;
        double thetaAcc = (gravity * sinTheta - cosTheta* temp) / (length * (4.0/3.0 - massPole * cosTheta * cosTheta / totalMass));
        double xAcc = temp - polemassLength * thetaAcc * cosTheta / totalMass;

        switch(kinematicsIntegrator) {
            case Euler:
                x += tau * xDot;
                xDot += tau * xAcc;
                theta += tau * thetaDot;
                thetaDot += tau * thetaAcc;
                break;

            case SemiImplicitEuler:
                xDot += tau * xAcc;
                x += tau * xDot;
                thetaDot += tau * thetaAcc;
                theta += tau * thetaDot;
                break;
        }

        episodeFinished |=  x < -xThreshold || x > xThreshold
                || theta < -thetaThresholdRadians || theta > thetaThresholdRadians;

        double reward;
        if(!episodeFinished) {
            reward = 1.0;
        }
        else if(stepsBeyondDone == null) {
            stepsBeyondDone = 0;
            reward = 1.0;
        }
        else {
            ++stepsBeyondDone;
            reward = 0;
        }

        Map<String, Object> channelsData = new HashMap<>() {{
            put("data", new double[]{x, xDot, theta, thetaDot});
        }};
        return new StepResult(channelsData, reward, episodeFinished);
    }

    @Override
    public void close() {
        // Do nothing
    }
}
