package org.arbiter.optimize.runner.listener.candidate;

import org.arbiter.optimize.runner.Status;
import org.arbiter.optimize.ui.ClientProvider;
import org.arbiter.optimize.ui.components.RenderElements;
import org.arbiter.optimize.ui.components.RenderableComponent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;

public class UICandidateStatusListenerImpl implements UICandidateStatusListener {

    private static final Logger log = LoggerFactory.getLogger(UICandidateStatusListener.class);
    private final int candidateNumber;
    private WebTarget target;

    public UICandidateStatusListenerImpl(int candidateNumber){
        this.candidateNumber = candidateNumber;
        //TODO don't hardcode
        target = ClientProvider.getClient().target("http://localhost:8080/modelResults/update/" + candidateNumber);
    }


    @Override
    public void reportStatus(Status status, RenderableComponent... uiElements) {
        target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                .post(Entity.entity(new RenderElements(uiElements), MediaType.APPLICATION_JSON));
        log.info("Update posted for candidate {}",candidateNumber);
    }
}
