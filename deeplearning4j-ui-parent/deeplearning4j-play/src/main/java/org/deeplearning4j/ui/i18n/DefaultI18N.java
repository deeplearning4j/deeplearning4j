package org.deeplearning4j.ui.i18n;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.ui.api.I18N;
import org.deeplearning4j.ui.api.UIModule;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

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
        loadLanguages();
    }

    private synchronized void loadLanguages(){
        ServiceLoader<UIModule> sl = ServiceLoader.load(UIModule.class);


        for(UIModule m : sl){
            List<File> resources = m.getInternationalizationResources();
            for(File f : resources){
                String path = f.getPath();
                int idxLast = path.lastIndexOf('.');
                if(idxLast < 0){
                    log.warn("Skipping language resource file: cannot infer language: {}", path);
                    continue;
                }

                String langCode = path.substring(idxLast+1).toLowerCase();
                Map<String,String> map = messagesByLanguage.computeIfAbsent(langCode, k -> new HashMap<>());

                parseFile(f, map);
            }
        }
    }

    private void parseFile(File file, Map<String,String> results){
        List<String> lines;
        try (FileInputStream fis = new FileInputStream(file)){
            lines = IOUtils.readLines(fis, Charset.forName("UTF-8"));
        } catch (IOException e){
            log.debug("Error parsing UI I18N content file; skipping: {}", file.getPath(), e.getMessage());
            return;
        }

        int count = 0;
        for (String line : lines) {
            if (!line.matches(".+=.*")) {
                log.debug("Invalid line in I18N file: {}, \"{}\"", file.getPath(), line);
                continue;
            }
            int idx = line.indexOf('=');
            String key = line.substring(0, idx);
            String value = line.substring(Math.min(idx + 1, line.length()));
            results.put(key, value);
            count++;
        }

        log.trace("Loaded {} messages from file {}", count, file.getPath());
    }

    @Override
    public String getMessage(String key) {
        return getMessage(currentLanguage, key);
    }

    @Override
    public String getMessage(String langCode, String key) {
        Map<String, String> messagesForLanguage = messagesByLanguage.get(langCode);

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
