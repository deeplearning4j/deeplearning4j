package org.deeplearning4j.nn.conf;

/**
 * Workspace mode to use. See https://deeplearning4j.org/workspaces<br>
 * <br>
 * NONE: No workspaces will be used for the network. Highest memory use, least performance.<br>
 * ENABLED: Use workspaces.<br>
 * SINGLE: Deprecated. Now equivalent to ENABLED, which should be used instead.<br>
 * SEPARATE: Deprecated. Now equivalent to ENABLED, which sohuld be used instead.<br>
 *
 * @author raver119@gmail.com
 */
public enum WorkspaceMode {
    NONE, // workspace won't be used
    ENABLED,
    /**
     * @deprecated Use {@link #ENABLED} instead
     */
    @Deprecated
    SINGLE, // one external workspace
    /**
     * @deprecated Use {@link #ENABLED} instead
     */
    @Deprecated
    SEPARATE, // one external workspace, one FF workspace, one BP workspace <-- default one
}
