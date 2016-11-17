#WEBSITE GUIDELINE

##PERMALINK

**DEFAULT ENGLISH** = deeplearning4j.org/**title** 

**OTHER LANGUAGE** = deeplearning4j.org/*lang*/**title**

**EXAMPLE:**

1. English  = deeplearning4j.org/**title**
2. Chinese  = deeplearning4j.org/*cn*/**title**
3. Korean   = deeplearning4j.org/*kr*/**title**
4. Japanese = deeplearning4j.org/*jp*/**title**

---

##WEBSITE STRUCTURE

The deeplearning4j.org website divided into the following sections:

1. Landing Page (deeplearning4j.org)
2. Content Page (deeplearning4j.org/xxx)
  * header
  * navigation
  * sidebar
  * footer

---

###Landing Page 

To edit the Landing Page, edit the html file located at **"_layouts/index.html"**

---

###Content Page

The default content layout is located at **"_layouts/default.html"**. The default layout contains html blocks from **"_includes/"**:

1. header.html
2. navigation.html
3. sidebar.html
4. footer.html


![WEBSITE LAYOUT](/img/website-layout.jpg)

To write a post with the layout you want, you should add this in the beginning of your post:

**---**

**title: YOUR TITLE**

**layout: default (or other layout you preferred)**

**---**

In example:

![MD LAYOUT](/img/sample-layout-theme.jpg)

---

###Header

Header html file located at **"_includes/header.html"**

**NOTE:** You shouldn't edit this file unless needed. 

---

###Navigation

Navigation html file located at **"_includes/navigation.html"**

If you are modifying the Navigation file for other language, then duplicated the **navigation.html** and rename it to **"lang-navigation.html"** (i.e. cn-navigation.html)

---

###Sidebar

Navigation html file located at **"_includes/sidebar.html"**

If you are modifying the Navigation file for other language, then duplicated the **navigation.html** and rename it to **"lang-sidebar.html"** (i.e. cn-sidebar.html)

---

###Footer

Footer html file located at **"_includes/footer.html"**

**NOTE:** You shouldn't edit this file unless needed. 



