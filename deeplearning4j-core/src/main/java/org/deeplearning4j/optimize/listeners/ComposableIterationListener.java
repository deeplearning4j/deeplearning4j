package org.deeplearning4j.optimize.listeners;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.ArrayList;
import java.util.Collection;

/**
 * A group of listeners
 * @author Adam Gibson
 */
public class ComposableIterationListener implements IterationListener {
    private Collection<IterationListener> listeners = new ArrayList<>();

    public ComposableIterationListener(Collection<IterationListener> listeners) {
        this.listeners = listeners;
    }

    @Override
    public void iterationDone(Model model,int iteration) {
        for(IterationListener listener : listeners)
            listener.iterationDone(model,iteration);
    }
}
