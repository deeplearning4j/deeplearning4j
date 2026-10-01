/*
 *  ******************************************************************************
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************/
package org.eclipse.deeplearning4j.vlm.model;

import org.eclipse.deeplearning4j.llm.config.ModelConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.ChatTemplate;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link VisionLanguageModel#generateChat} pairs every image part with one supplied image in
 * conversation order; a conversation that disagrees with its images must fail before any
 * graph runs rather than fill image slots out of order.
 */
public class VisionLanguageModelChatValidationTest {

    private static VisionLanguageModel emptyModel() {
        return VisionLanguageModel.builder().config(new ModelConfig()).build();
    }

    @Test
    public void imagePartWithoutImageIsRejected() {
        List<ChatTemplate.Message> messages = List.of(new ChatTemplate.Message("user",
                List.of(ChatTemplate.ContentPart.image(), ChatTemplate.ContentPart.text("What is this?"))));

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> emptyModel().generateChat(messages, List.of(), null));
        assertTrue(error.getMessage().contains("1 image part(s) but 0 image(s)"), error.getMessage());
    }

    @Test
    public void imageWithoutImagePartIsRejected() {
        List<ChatTemplate.Message> messages = List.of(ChatTemplate.Message.user("Describe it"));
        BufferedImage image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> emptyModel().generateChat(messages, List.of(image), null));
        assertTrue(error.getMessage().contains("0 image part(s) but 1 image(s)"), error.getMessage());
    }

    @Test
    public void imagePartsAcrossTurnsAreCountedTogether() {
        List<ChatTemplate.Message> messages = List.of(
                new ChatTemplate.Message("user", List.of(ChatTemplate.ContentPart.image())),
                ChatTemplate.Message.assistant("A cat."),
                new ChatTemplate.Message("user",
                        List.of(ChatTemplate.ContentPart.image(), ChatTemplate.ContentPart.text("And this?"))));
        BufferedImage image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> emptyModel().generateChat(messages, List.of(image), null));
        assertTrue(error.getMessage().contains("2 image part(s) but 1 image(s)"), error.getMessage());
    }

    @Test
    public void emptyConversationIsRejected() {
        assertThrows(IllegalArgumentException.class,
                () -> emptyModel().generateChat(List.of(), List.of(), null));
    }
}
