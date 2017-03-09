package org.deeplearning4j.ui.i18n;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.api.I18N;
import org.reflections.Reflections;
import org.reflections.scanners.ResourcesScanner;
import org.reflections.util.ConfigurationBuilder;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;
import java.util.regex.Pattern;

/**
 * Default internationalization implementation.<br>
 * Content for internationalization is implemented using resource files.<br>
 * For the resource files: they should be specified as follows:<br>
 * 1. In the /dl4j_i18n/ directory in resources<br>
 * 2. Filenames should be "somekey.langcode" - for example, "index.en" or "index.ja"<br>
 * 3. Each key should be unique across all files. Any key can appear in any file; files may be split for convenience<br>
 * <p>
 * Loading of these UI resources is done as follows:<br>
 * - On initialization of the DefaultI18N:<br>
 * &nbsp;&nbsp;- Resource files for the default language are loaded<br>
 * - If a different language is requested, the content will be loaded on demand (and stored in memory for future use)<br>
 * Note that if a specified language does not have the specified key, the result from the defaultfallback language (English)
 * will be used instead.
 *
 * @author Alex Black
 */
@Slf4j
public class DefaultI18N implements I18N {

    public static final String DEFAULT_LANGUAGE = "en";
    public static final String FALLBACK_LANGUAGE = "en"; //use this if the specified language doesn't have the requested message
    public static final String DEFAULT_I8N_RESOURCES_DIR = "dl4j_i18n";

    private static DefaultI18N instance;

    private Map<String, Map<String, String>> messagesByLanguage = new HashMap<>();

    public static synchronized I18N getInstance() {
        if (instance == null)
            instance = new DefaultI18N();
        return instance;
    }


    private String currentLanguage = DEFAULT_LANGUAGE;

    private Set<String> loadedLanguages = Collections.synchronizedSet(new HashSet<>());

    private DefaultI18N() {
        //Load default language...
        loadLanguageResources(currentLanguage);
    }

    private synchronized void loadLanguageResources(String languageCode) {
        if (loadedLanguages.contains(languageCode))
            return;

        //Scan classpath for resources in the /dl4j_i18n/ directory...
        URL url = this.getClass().getResource("/" + DEFAULT_I8N_RESOURCES_DIR + "/");
        Reflections reflections =
                        new Reflections(new ConfigurationBuilder().setScanners(new ResourcesScanner()).setUrls(url));

        String pattern = ".*" + languageCode;
        Set<String> resources = reflections.getResources(Pattern.compile(pattern));

        Map<String, String> messages = new HashMap<>();

        for (String s : resources) {
            if (!s.endsWith(languageCode))
                continue;

            log.trace("Attempting to parse file: {}", s);
            parseFile(s, messages);
        }

        messagesByLanguage.put(languageCode, messages);

        loadedLanguages.add(languageCode);
    }

    private void parseFile(String filename, Map<String, String> results) {

        List<String> lines;
        try {
            String path;
            if (filename.startsWith(DEFAULT_I8N_RESOURCES_DIR)) {
                //As a resource from JAR file - already has dir at the start...
                path = "/" + filename;
            } else {
                //Run in dev environment - no dir at the start...
                path = "/" + DEFAULT_I8N_RESOURCES_DIR + "/" + filename;
            }
            InputStream is = this.getClass().getResourceAsStream(path);
            lines = IOUtils.readLines(is);
        } catch (Exception e) {
            log.debug("Error parsing UI I18N content file; skipping: {}", filename, e.getMessage());
            return;
        }

        //TODO need to think more carefully about how to parse this, with multi-line messages, etc
        int count = 0;
        for (String line : lines) {
            if (!line.matches(".+=.*")) {
                log.debug("Invalid line in I18N file: {}, \"{}\"", filename, line);
                continue;
            }
            int idx = line.indexOf('=');
            String key = line.substring(0, idx);
            String value = line.substring(Math.min(idx + 1, line.length()));
            results.put(key, value);
            count++;
        }

        log.trace("Loaded {} messages from file {}", count, filename);
    }

    @Override
    public String getMessage(String key) {
        return getMessage(currentLanguage, key);
    }

    @Override
    public String getMessage(String langCode, String key) {
        Map<String, String> messagesForLanguage = messagesByLanguage.get(langCode);
        if (messagesForLanguage == null) {
            synchronized (this) {
                //Synchronized to avoid loading multiple times in case of multi-threaded requests
                if (messagesByLanguage.get(langCode) == null) {
                    loadLanguageResources(langCode);
                }
            }
            messagesForLanguage = messagesByLanguage.get(langCode);
        }

        String msg = messagesForLanguage.get(key);
        if (msg == null && !FALLBACK_LANGUAGE.equals(langCode)) {
            //Try getting the result from the fallback language
            return getMessage(FALLBACK_LANGUAGE, key);
        }

        return msg;
    }

    @Override
    public String getDefaultLanguage() {
        return currentLanguage;
    }

    @Override
    public void setDefaultLanguage(String langCode) {
        this.currentLanguage = langCode;
        log.debug("UI: Set language to {}", langCode);
    }
}
