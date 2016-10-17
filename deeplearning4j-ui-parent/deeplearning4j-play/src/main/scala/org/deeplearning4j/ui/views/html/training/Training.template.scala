
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Training_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Training extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<html>
    <head>
        <link rel="stylesheet" type="text/css" href="assets/css/train.css"/><meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
        <title>"""),_display_(/*5.17*/i18n/*5.21*/.getMessage("train.pagetitle")),format.raw/*5.51*/("""</title>
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
                <div class="topBarDivContent">Deeplearning4j UI</div>
            </div>

            <div class="navDiv">
                <div class="navTopSpacer"></div>
                <div class="navSessionID">
                    <strong>"""),_display_(/*31.30*/i18n/*31.34*/.getMessage("train.nav.sessionid")),format.raw/*31.68*/(""":</strong><br>
                </div>
                <div class="navSessionID" id="navSessionIDValue">
                    -
                </div>
                <div class="navTopSpacer"></div>
                <div class="navElement" id="overviewNavDiv" onclick="onNavClick('overview', '"""),_display_(/*37.95*/i18n/*37.99*/.getMessage("train.nav.errormsg")),format.raw/*37.132*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_home_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*41.22*/i18n/*41.26*/.getMessage("train.nav.overview")),format.raw/*41.59*/("""
                """),format.raw/*42.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" id="modelNavDiv" onclick="onNavClick('model', '"""),_display_(/*44.89*/i18n/*44.93*/.getMessage("train.nav.errormsg")),format.raw/*44.126*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_model_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*48.22*/i18n/*48.26*/.getMessage("train.nav.model")),format.raw/*48.56*/("""
                """),format.raw/*49.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" id="systemNavDiv" onclick="onNavClick('system', '"""),_display_(/*51.91*/i18n/*51.95*/.getMessage("train.nav.errormsg")),format.raw/*51.128*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_system_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*55.22*/i18n/*55.26*/.getMessage("train.nav.hwsw")),format.raw/*55.55*/("""
                """),format.raw/*56.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" id="helpNavDiv" onclick="onNavClick('help', '"""),_display_(/*58.87*/i18n/*58.91*/.getMessage("train.nav.errormsg")),format.raw/*58.124*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_help_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*62.22*/i18n/*62.26*/.getMessage("train.nav.help")),format.raw/*62.55*/("""
                """),format.raw/*63.17*/("""</div>
                <div class="navBottom">
                    """),_display_(/*65.22*/i18n/*65.26*/.getMessage("train.nav.language")),format.raw/*65.59*/(""":
                    <select class="languageSelect" id="navLangSelect" onchange="changeLanguage('"""),_display_(/*66.98*/i18n/*66.102*/.getMessage("train.nav.langChangeErrorMsg")),format.raw/*66.145*/("""')">
                        <option value="en">English</option>
                        <option value="ja">日本語</option>
                        <option value="zh">中文</option>
                        <option value="kr">한글</option>
                    </select>
                    <div class="navElementSpacer"></div>
                    <a href=""""),_display_(/*73.31*/i18n/*73.35*/.getMessage("train.nav.websitelink")),format.raw/*73.71*/("""" class="textlink">"""),_display_(/*73.91*/i18n/*73.95*/.getMessage("train.nav.websitelinktext")),format.raw/*73.135*/("""</a>
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
                  DATE: Mon Oct 17 11:06:45 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: 3130ff3ed4041696b9d0607e48a17ce019870f26
                  MATRIX: 588->1|721->39|749->41|955->221|967->225|1017->255|1385->595|1414->596|1456->610|1508->635|1521->639|1575->672|1748->818|1777->819|1810->825|2323->1311|2336->1315|2391->1349|2716->1647|2729->1651|2784->1684|3031->1904|3044->1908|3098->1941|3144->1959|3321->2109|3334->2113|3389->2146|3637->2367|3650->2371|3701->2401|3747->2419|3926->2571|3939->2575|3994->2608|4243->2830|4256->2834|4306->2863|4352->2881|4527->3029|4540->3033|4595->3066|4842->3286|4855->3290|4905->3319|4951->3337|5048->3407|5061->3411|5115->3444|5242->3544|5256->3548|5321->3591|5703->3946|5716->3950|5773->3986|5820->4006|5833->4010|5895->4050
                  LINES: 20->1|25->1|26->2|29->5|29->5|29->5|37->13|37->13|38->14|38->14|38->14|38->14|42->18|42->18|43->19|55->31|55->31|55->31|61->37|61->37|61->37|65->41|65->41|65->41|66->42|68->44|68->44|68->44|72->48|72->48|72->48|73->49|75->51|75->51|75->51|79->55|79->55|79->55|80->56|82->58|82->58|82->58|86->62|86->62|86->62|87->63|89->65|89->65|89->65|90->66|90->66|90->66|97->73|97->73|97->73|97->73|97->73|97->73
                  -- GENERATED --
              */
          