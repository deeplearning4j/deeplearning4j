package org.nd4j.samediff.frameworkimport.reflect

import org.apache.commons.io.FileUtils
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Test
import java.io.File
import java.nio.charset.Charset
import kotlin.test.assertTrue


class ClassGraphHolderTest {

    @Test
    fun testClassGraphHolder() {
        val jsonFile = File("scanned-classes.json")
        ClassGraphHolder.saveScannedClasses(jsonFile)
        val original = ClassGraphHolder.scannedClasses
        val loadedJson = FileUtils.readFileToString(jsonFile, Charset.defaultCharset())
        assertTrue(loadedJson.length > 1,"Json was not written and is empty")
        val loaded = ClassGraphHolder.loadFromJson(loadedJson)
        assertEquals(original.toJSON(),loaded.toJSON())
        assertNotNull(ClassGraphHolder.scannedClasses)
    }

}