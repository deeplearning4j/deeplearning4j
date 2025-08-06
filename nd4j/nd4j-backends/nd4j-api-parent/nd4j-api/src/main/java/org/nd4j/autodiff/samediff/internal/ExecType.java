package org.nd4j.autodiff.samediff.internal;

/**
 * ExecType: Execution type, as used in ExecStep<br>
 * OP: Operation execution<br>
 * VARIABLE: Variable "execution", mainly used to trigger ops that depend on the
 * variable<br>
 * CONSTANT: As per variable<br>
 * PLACEHOLDER: As per variable<br>
 * SWITCH_L and SWITCH_R: This is a bit of a hack to account for the fact that
 * only one of
 * the switch branches (left or right) will ever be available; without this,
 * once the switch op is executed, we'll
 * (incorrectly) conclude that *both* branches can be executed<br>
 * EXEC_START: Start of execution<br>
 * CONTROL_DEP: Control dependency for op. Used for TF import, due to its odd
 * "constant depends on op in a frame" behaviour
 */
public enum ExecType {
    OP, VARIABLE, CONSTANT, PLACEHOLDER, SWITCH_L, SWITCH_R, EXEC_START, CONTROL_DEP
}
