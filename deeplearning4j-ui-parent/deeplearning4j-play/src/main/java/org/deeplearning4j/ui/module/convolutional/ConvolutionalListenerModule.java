package org.deeplearning4j.ui.module.convolutional;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.Persistable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.api.storage.StatsStorageEvent;
import org.deeplearning4j.api.storage.StatsStorageListener;
import org.deeplearning4j.ui.api.FunctionType;
import org.deeplearning4j.ui.api.HttpMethod;
import org.deeplearning4j.ui.api.Route;
import org.deeplearning4j.ui.api.UIModule;
import org.deeplearning4j.ui.i18n.I18NResource;
import org.deeplearning4j.ui.weights.ConvolutionListenerPersistable;
import play.mvc.Result;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import static play.mvc.Results.ok;

/**
 * Used for plotting results from the ConvolutionalIterationListener
 *
 * @author Alex Black
 */
@Slf4j
public class ConvolutionalListenerModule implements UIModule {

    private static final String TYPE_ID = "ConvolutionalListener";

    private StatsStorage lastStorage;
    private String lastSessionID;
    private String lastWorkerID;
    private long lastTimeStamp;

    @Override
    public List<String> getCallbackTypeIDs() {
        return Collections.singletonList(TYPE_ID);
    }

    @Override
    public List<Route> getRoutes() {
        Route r = new Route("/activations", HttpMethod.GET, FunctionType.Supplier,
                        () -> ok(org.deeplearning4j.ui.views.html.convolutional.Activations.apply()));
        Route r2 = new Route("/activations/data", HttpMethod.GET, FunctionType.Supplier, this::getImage);

        return Arrays.asList(r, r2);
    }

    @Override
    public synchronized void reportStorageEvents(Collection<StatsStorageEvent> events) {
        for (StatsStorageEvent sse : events) {
            if (TYPE_ID.equals(sse.getTypeID())
                            && sse.getEventType() == StatsStorageListener.EventType.PostStaticInfo) {
                if (sse.getTimestamp() > lastTimeStamp) {
                    lastStorage = sse.getStatsStorage();
                    lastSessionID = sse.getSessionID();
                    lastWorkerID = sse.getWorkerID();
                    lastTimeStamp = sse.getTimestamp();
                }
            }
        }
    }

    @Override
    public void onAttach(StatsStorage statsStorage) {

    }

    @Override
    public void onDetach(StatsStorage statsStorage) {

    }

    @Override
    public List<I18NResource> getInternationalizationResources() {
        return Collections.emptyList();
    }

    private Result getImage() {
        if (lastTimeStamp > 0 && lastStorage != null) {
            Persistable p = lastStorage.getStaticInfo(lastSessionID, TYPE_ID, lastWorkerID);
            if (p instanceof ConvolutionListenerPersistable) {
                ConvolutionListenerPersistable clp = (ConvolutionListenerPersistable) p;
                BufferedImage bi = clp.getImg();
                ByteArrayOutputStream baos = new ByteArrayOutputStream();
                try {
                    ImageIO.write(bi, "jpg", baos);
                } catch (IOException e) {
                    log.warn("Error displaying image", e);
                }
                return ok(baos.toByteArray()).as("image/jpg");
            } else {
                return ok(new byte[0]).as("image/jpeg");
            }
        } else {
            return ok(new byte[0]).as("image/jpeg");
        }
    }
}
