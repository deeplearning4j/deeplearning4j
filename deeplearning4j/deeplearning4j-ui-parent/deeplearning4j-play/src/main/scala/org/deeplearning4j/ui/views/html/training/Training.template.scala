
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._


     object Training_Scope0 {

class Training extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<html>
<head>
    <link rel="stylesheet" type="text/css" href="assets/css/train.css"/><meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
    <title>"""),_display_(/*5.13*/i18n/*5.17*/.getMessage("train.pagetitle")),format.raw/*5.47*/("""</title>
    <!-- jQuery, D3.js, etc -->
    <script src="/assets/jquery-2.2.0.min.js"></script>
    <script src="/assets/notify.js"></script>
    <!-- Custom assets compiled from Typescript -->
    <script src="/assets/js/dl4j-play-ui.js"></script>
</head>
<script>
        function onPageLoad()"""),format.raw/*13.30*/("""{"""),format.raw/*13.31*/("""
            """),format.raw/*14.13*/("""onNavClick('overview', '"""),_display_(/*14.38*/i18n/*14.42*/.getMessage("train.nav.errormsg")),format.raw/*14.75*/("""');
            setSessionIDDivContents();
            setLanguageSelectorValue();
            setInterval(getAndProcessUpdate,2000);
        """),format.raw/*18.9*/("""}"""),format.raw/*18.10*/("""
    """),format.raw/*19.5*/("""</script>
<body onload="onPageLoad()">

<div class="outerDiv">
    <div class="topBarDiv">
        <div class="topBarDivContent"><a href="/"><img src="/assets/deeplearning4j.img" border="0"/></a></div>
        <div class="topBarDivContent">Deeplearning4j UI v1</div>
    </div>

    <div class="navDiv">
        <div class="navTopSpacer"></div>
        <div class="navSessionID">
            <strong>"""),_display_(/*31.22*/i18n/*31.26*/.getMessage("train.nav.sessionid")),format.raw/*31.60*/(""":</strong><br>
        </div>
        <div class="navSessionID" id="navSessionIDValue">
            -
        </div>
        <div class="navTopSpacer"></div>
        <div class="navElement" id="overviewNavDiv" onclick="onNavClick('overview', '"""),_display_(/*37.87*/i18n/*37.91*/.getMessage("train.nav.errormsg")),format.raw/*37.124*/("""')">
        <div class="navHorizontalSpacer"></div>
        <img src="assets/img/train/nav_icon_home_28.png">
        <div class="navHorizontalSpacer"></div>
        """),_display_(/*41.10*/i18n/*41.14*/.getMessage("train.nav.overview")),format.raw/*41.47*/("""
    """),format.raw/*42.5*/("""</div>
    <div class="navElementSpacer"></div>
    <div class="navElement" id="modelNavDiv" onclick="onNavClick('model', '"""),_display_(/*44.77*/i18n/*44.81*/.getMessage("train.nav.errormsg")),format.raw/*44.114*/("""')">
    <div class="navHorizontalSpacer"></div>
    <img src="assets/img/train/nav_icon_model_28.png">
    <div class="navHorizontalSpacer"></div>
    """),_display_(/*48.6*/i18n/*48.10*/.getMessage("train.nav.model")),format.raw/*48.40*/("""
"""),format.raw/*49.1*/("""</div>
<div class="navElementSpacer"></div>
<div class="navElement" id="systemNavDiv" onclick="onNavClick('system', '"""),_display_(/*51.75*/i18n/*51.79*/.getMessage("train.nav.errormsg")),format.raw/*51.112*/("""')">
<div class="navHorizontalSpacer"></div>
<img src="assets/img/train/nav_icon_system_28.png">
<div class="navHorizontalSpacer"></div>
"""),_display_(/*55.2*/i18n/*55.6*/.getMessage("train.nav.hwsw")),format.raw/*55.35*/("""
"""),format.raw/*56.1*/("""</div>
<div class="navElementSpacer"></div>
<div class="navElement" id="helpNavDiv" onclick="onNavClick('help', '"""),_display_(/*58.71*/i18n/*58.75*/.getMessage("train.nav.errormsg")),format.raw/*58.108*/("""')">
<div class="navHorizontalSpacer"></div>
<img src="assets/img/train/nav_icon_help_28.png">
<div class="navHorizontalSpacer"></div>
"""),_display_(/*62.2*/i18n/*62.6*/.getMessage("train.nav.help")),format.raw/*62.35*/("""
"""),format.raw/*63.1*/("""</div>
<div class="navBottom">
    """),_display_(/*65.6*/i18n/*65.10*/.getMessage("train.nav.language")),format.raw/*65.43*/(""":
    <select class="languageSelect" id="navLangSelect" onchange="changeLanguage('"""),_display_(/*66.82*/i18n/*66.86*/.getMessage("train.nav.langChangeErrorMsg")),format.raw/*66.129*/("""')">
    <option value="en">English</option>
    <option value="ja">日本語</option>
    <option value="zh">中文</option>
    <option value="kr">한글</option>
    </select>
    <div class="navElementSpacer"></div>
    <a href=""""),_display_(/*73.15*/i18n/*73.19*/.getMessage("train.nav.websitelink")),format.raw/*73.55*/("""" class="textlink">"""),_display_(/*73.75*/i18n/*73.79*/.getMessage("train.nav.websitelinktext")),format.raw/*73.119*/("""</a>
    <div class="navElementSpacer"></div>
</div>
</div>


<div class="contentDiv" id="mainContentDiv">

</div>
</div>
</body>
</html>"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object Training extends Training_Scope0.Training
              /*
                  -- GENERATED --
                  DATE: Sun Oct 23 21:43:02 PDT 2016
                  SOURCE: /Users/ejunprung/skymind-ui/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: bf36c6ed95b56f40d25975657bde7eb5b898b648
                  MATRIX: 588->1|721->39|748->40|939->205|951->209|1001->239|1325->535|1354->536|1395->549|1447->574|1460->578|1514->611|1683->753|1712->754|1744->759|2172->1160|2185->1164|2240->1198|2511->1442|2524->1446|2579->1479|2774->1647|2787->1651|2841->1684|2873->1689|3024->1813|3037->1817|3092->1850|3271->2003|3284->2007|3335->2037|3363->2038|3508->2156|3521->2160|3576->2193|3740->2331|3752->2335|3802->2364|3830->2365|3971->2479|3984->2483|4039->2516|4201->2652|4213->2656|4263->2685|4291->2686|4353->2722|4366->2726|4420->2759|4530->2842|4543->2846|4608->2889|4855->3109|4868->3113|4925->3149|4972->3169|4985->3173|5047->3213
                  LINES: 20->1|25->1|26->2|29->5|29->5|29->5|37->13|37->13|38->14|38->14|38->14|38->14|42->18|42->18|43->19|55->31|55->31|55->31|61->37|61->37|61->37|65->41|65->41|65->41|66->42|68->44|68->44|68->44|72->48|72->48|72->48|73->49|75->51|75->51|75->51|79->55|79->55|79->55|80->56|82->58|82->58|82->58|86->62|86->62|86->62|87->63|89->65|89->65|89->65|90->66|90->66|90->66|97->73|97->73|97->73|97->73|97->73|97->73
                  -- GENERATED --
              */
          