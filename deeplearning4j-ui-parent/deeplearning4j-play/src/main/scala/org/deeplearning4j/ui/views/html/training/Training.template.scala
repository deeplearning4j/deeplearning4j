
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
    <body>

        <div class="outerDiv">
            <div class="topBarDiv">
                <div class="topBarDivContent"><a href="/"><img src="/assets/deeplearning4j.img" border="0"/></a></div>
                <div class="topBarDivContent">Deeplearning4j UI</div>
            </div>

            <div class="navDiv">
                <div class="navTopSpacer">(TODO: Session Selection Here)</div>
                <div class="navElement" onclick="onNavClick('home', '"""),_display_(/*22.71*/i18n/*22.75*/.getMessage("train.nav.errormsg")),format.raw/*22.108*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_home_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*26.22*/i18n/*26.26*/.getMessage("train.nav.overview")),format.raw/*26.59*/("""
                """),format.raw/*27.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" onclick="onNavClick('model', '"""),_display_(/*29.72*/i18n/*29.76*/.getMessage("train.nav.errormsg")),format.raw/*29.109*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_model_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*33.22*/i18n/*33.26*/.getMessage("train.nav.model")),format.raw/*33.56*/("""
                """),format.raw/*34.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" onclick="onNavClick('system', '"""),_display_(/*36.73*/i18n/*36.77*/.getMessage("train.nav.errormsg")),format.raw/*36.110*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_system_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*40.22*/i18n/*40.26*/.getMessage("train.nav.hwsw")),format.raw/*40.55*/("""
                """),format.raw/*41.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" onclick="onNavClick('help', '"""),_display_(/*43.71*/i18n/*43.75*/.getMessage("train.nav.errormsg")),format.raw/*43.108*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_help_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*47.22*/i18n/*47.26*/.getMessage("train.nav.help")),format.raw/*47.55*/("""
                """),format.raw/*48.17*/("""</div>
                <div class="navBottom">
                    (TODO Language Selection)
                    <div class="navElementSpacer"></div>
                    <a href="http://www.deeplearning4j.org/" class="textlink">deeplearning4j.org</a>
                    <div class="navElementSpacer"></div>
                </div>
            </div>
            <div class="contentDiv" id="mainContentDiv">
                Content Div<br>
                Language: """),_display_(/*58.28*/i18n/*58.32*/.getDefaultLanguage),format.raw/*58.51*/("""
            """),format.raw/*59.13*/("""</div>
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
                  DATE: Sat Oct 15 15:56:16 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: eb9937724213107a7bf1c0b4f3310943b6ea52ba
                  MATRIX: 588->1|721->39|749->41|955->221|967->225|1017->255|1821->1032|1834->1036|1889->1069|2136->1289|2149->1293|2203->1326|2249->1344|2409->1477|2422->1481|2477->1514|2725->1735|2738->1739|2789->1769|2835->1787|2996->1921|3009->1925|3064->1958|3313->2180|3326->2184|3376->2213|3422->2231|3581->2363|3594->2367|3649->2400|3896->2620|3909->2624|3959->2653|4005->2671|4508->3147|4521->3151|4561->3170|4603->3184
                  LINES: 20->1|25->1|26->2|29->5|29->5|29->5|46->22|46->22|46->22|50->26|50->26|50->26|51->27|53->29|53->29|53->29|57->33|57->33|57->33|58->34|60->36|60->36|60->36|64->40|64->40|64->40|65->41|67->43|67->43|67->43|71->47|71->47|71->47|72->48|82->58|82->58|82->58|83->59
                  -- GENERATED --
              */
          