package org.arbiter.optimize.runner.listener.candidate;

import org.arbiter.optimize.runner.Status;
import org.arbiter.optimize.ui.components.RenderableComponent;

/**Listener for reporting candidate status
 */
public interface UICandidateStatusListener {

    void reportStatus(Status status, RenderableComponent... uiComponents);

}
