package org.deeplearning4j.nn.conf;

/**
 * @author raver119@gmail.com
 */
public enum WorkspaceMode {
    NONE, // workspace won't be used
    SINGLE, // one external workspace
    SEPARATE, // one external workspace, one FF workspace, one BP workspace <-- default one
}
