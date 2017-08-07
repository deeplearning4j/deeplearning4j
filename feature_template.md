Here is a template for creating a feature page. The goal is to apply it to both new and existing features, fitting any extant documentation into this mold.

Restrict content to the feature at hand; broader, more conceptual discussions have their own place.

### NB. Whenever you use a term that has documentation associated with it elsewhere, please link to that documentation.

# Name of Feature

## Table of Contents

A table of contents with hyperlinks to each of the following **headings** and **subheadings**.

## Description

A brief overview of the feature, such as might be included in a glossary or presentation (between 1 sentence and 1-2 paragraphs in length).

A complex feature (e.g., Datavec) may be accompanied by a longer description, broken down into a series of **subheadings**.

## Examples and Use Cases

Code examples identified according to their real-world applications (e.g., image recognition, prediction, etc.).
Document each example as a **subheading**, included in the main table of contents.

## How It Works

Think of this section as a recipe for deployment:

### Prerequisites
   These are your ingredients, particularly other DL4J features from which you will assemble this feature (e.g., DataVec requires a sentence iterator, tokenizer, etc.).
   
followed by 

### Step-by-step instructions
   Present steps in the most logical, effecient, user-friendly sequence.
   Embed relevant code at each step.
   
Begin the section with a section-specific table of contents to aid navigation. Include the following as subheadings in the main table of contents as well:
### 0. Prerequisites
### 1. Phrase/sentence summarizing first step
### 2. Phrase/sentence summarizing second step
etc.

A well-written recipe presents ingredients in the order in which they will be used. Once you've laid out the steps in the best order possible, confirm that your prerequisite are ordered accordingly.

At the end of the setup, include a link to the feature **javadoc**.

## Troubleshooting

Q&A for fixing common problems. Feel free to include material from the DL4J channel or come up with your own.

## Disclaimers

Any important provisos (e.g., patents) would go here.

## Further reading

Books, articles, links to other documents within our libraries or others.






