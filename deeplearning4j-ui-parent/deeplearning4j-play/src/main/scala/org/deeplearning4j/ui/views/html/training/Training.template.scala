
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
        <title>Document</title>
    </head>
    <body>

        <div class="outerDiv">

            <div class="navDiv">
                <div class="navTopSpacer">(TODO: Session Selection Here)</div>
                <div class="navElement">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_home_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*17.22*/i18n/*17.26*/.getMessage("train.nav.overview")),format.raw/*17.59*/("""
                """),format.raw/*18.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_model_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*24.22*/i18n/*24.26*/.getMessage("train.nav.model")),format.raw/*24.56*/("""
                """),format.raw/*25.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_system_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*31.22*/i18n/*31.26*/.getMessage("train.nav.hwsw")),format.raw/*31.55*/("""
                """),format.raw/*32.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_help_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*38.22*/i18n/*38.26*/.getMessage("train.nav.help")),format.raw/*38.55*/("""
                """),format.raw/*39.17*/("""</div>
                <div class="navBottom">
                    (TODO Language Selection Here)
                    <div class="navElementSpacer"></div>
                    <a href="http://www.deeplearning4j.org/" class="textlink">deeplearning4j.org</a>
                    <div class="navElementSpacer"></div>
                </div>
            </div>
            <div class="contentDiv">
                Content Div<br>
                Language: """),_display_(/*49.28*/i18n/*49.32*/.getDefaultLanguage),format.raw/*49.51*/("""
            """),format.raw/*50.13*/("""</div>
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
                  DATE: Sat Oct 15 11:35:46 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: 70dd1f57d6d6b31e6b7ba18fa6c8c9def42f906c
                  MATRIX: 588->1|721->39|749->41|1404->669|1417->673|1471->706|1517->724|1863->1043|1876->1047|1927->1077|1973->1095|2320->1415|2333->1419|2383->1448|2429->1466|2774->1784|2787->1788|2837->1817|2883->1835|3371->2296|3384->2300|3424->2319|3466->2333
                  LINES: 20->1|25->1|26->2|41->17|41->17|41->17|42->18|48->24|48->24|48->24|49->25|55->31|55->31|55->31|56->32|62->38|62->38|62->38|63->39|73->49|73->49|73->49|74->50
                  -- GENERATED --
              */
          