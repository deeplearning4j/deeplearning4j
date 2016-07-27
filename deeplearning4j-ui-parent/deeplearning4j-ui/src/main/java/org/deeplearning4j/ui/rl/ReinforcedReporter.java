package org.deeplearning4j.ui.rl;

import com.fasterxml.jackson.jaxrs.json.JacksonJsonProvider;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.deeplearning4j.ui.providers.ObjectMapperProvider;
import org.deeplearning4j.ui.rl.beans.ReportBean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.Entity;
import javax.ws.rs.client.WebTarget;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

/**
 * @author raver119@gmail.com
 */
public class ReinforcedReporter {
    private static Logger log = LoggerFactory.getLogger(ReinforcedReporter.class);
    private final UiConnectionInfo connectionInfo;

    private Client client = ClientBuilder.newClient().register(JacksonJsonProvider.class).register(new ObjectMapperProvider());
    private WebTarget target;

    public ReinforcedReporter(UiConnectionInfo connectionInfo) {
        this.connectionInfo = connectionInfo;

        target = client.target(connectionInfo.getFirstPart()).path(connectionInfo.getSecondPart("rl")).path("state").queryParam("sid", connectionInfo.getSessionId());
    }

    public void report(long epochId, double reward) {

        ReportBean bean = new ReportBean(epochId, reward);

        Response resp = target.request(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON).post(Entity.entity(bean, MediaType.APPLICATION_JSON));
        log.debug("Response: " + resp);
    }


}
