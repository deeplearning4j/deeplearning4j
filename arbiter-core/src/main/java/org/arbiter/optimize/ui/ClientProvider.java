package org.arbiter.optimize.ui;

import com.fasterxml.jackson.jaxrs.json.JacksonJsonProvider;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;

public class ClientProvider {

    private static Client instance = ClientBuilder.newClient().register(JacksonJsonProvider.class);

    private ClientProvider(){ }

    public static Client getClient(){
        return instance;
    }

}
