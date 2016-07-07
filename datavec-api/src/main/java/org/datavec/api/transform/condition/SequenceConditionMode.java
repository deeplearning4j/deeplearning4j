package org.datavec.api.transform.condition;

/**
 * For certain single-column conditions: how should we apply these to sequences?<br>
 * <b>And</b>: Condition applies to sequence only if it applies to ALL time steps<br>
 * <b>Or</b>: Condition applies to sequence if it applies to ANY time steps<br>
 * <b>NoSequencMode</b>: Condition cannot be applied to sequences at all (error condition)
 *
 * @author Alex Black
 */
public enum SequenceConditionMode {
    And, Or, NoSequenceMode
}
