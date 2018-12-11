package org.nd4j.autodiff.samediff;

/**
 * An SDVarible may have different uses in a SameDiff graph - VariableType represents these different roles/uses.<br>
 * <br>
 * VARIABLE: a trainable parameter in the SameDiff instance.<br>
 * CONSTANT: a fixed value that should not be modified during training/inference. May be replaced by the user.<br>
 * ARRAY: an intermediate array that is not trainable and is usually not persisted. For example, activations.<br>
 * <br>
 * <br>
 * <pre>
 * Type         Trainable   Gradients       Persisted       Workspaces
 * VARIABLE     Yes         Yes             Yes             No
 * CONSTANT     No          No              Yes             No
 * ARRAY        No          Yes             No              Yes
 * PLACEHOLDER  No          No              No              No
 * </pre>
 *
 */
public enum VariableType {
    VARIABLE,
    CONSTANT,
    ARRAY,
    PLACEHOLDER
}
