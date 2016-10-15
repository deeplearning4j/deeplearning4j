
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
            """),format.raw/*14.13*/("""onNavClick('home', '"""),_display_(/*14.34*/i18n/*14.38*/.getMessage("train.nav.errormsg")),format.raw/*14.71*/("""');
            setLanguageSelectorValue();
        """),format.raw/*16.9*/("""}"""),format.raw/*16.10*/("""
    """),format.raw/*17.5*/("""</script>
    <body onload="onPageLoad()">

        <div class="outerDiv">
            <div class="topBarDiv">
                <div class="topBarDivContent"><a href="/"><img src="/assets/deeplearning4j.img" border="0"/></a></div>
                <div class="topBarDivContent">Deeplearning4j UI</div>
            </div>

            <div class="navDiv">
                <div class="navTopSpacer">(TODO: Session Selection Here)</div>
                <div class="navElement" id="homeNavDiv" onclick="onNavClick('home', '"""),_display_(/*28.87*/i18n/*28.91*/.getMessage("train.nav.errormsg")),format.raw/*28.124*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_home_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*32.22*/i18n/*32.26*/.getMessage("train.nav.overview")),format.raw/*32.59*/("""
                """),format.raw/*33.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" id="modelNavDiv" onclick="onNavClick('model', '"""),_display_(/*35.89*/i18n/*35.93*/.getMessage("train.nav.errormsg")),format.raw/*35.126*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_model_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*39.22*/i18n/*39.26*/.getMessage("train.nav.model")),format.raw/*39.56*/("""
                """),format.raw/*40.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" id="systemNavDiv" onclick="onNavClick('system', '"""),_display_(/*42.91*/i18n/*42.95*/.getMessage("train.nav.errormsg")),format.raw/*42.128*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_system_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*46.22*/i18n/*46.26*/.getMessage("train.nav.hwsw")),format.raw/*46.55*/("""
                """),format.raw/*47.17*/("""</div>
                <div class="navElementSpacer"></div>
                <div class="navElement" id="helpNavDiv" onclick="onNavClick('help', '"""),_display_(/*49.87*/i18n/*49.91*/.getMessage("train.nav.errormsg")),format.raw/*49.124*/("""')">
                    <div class="navHorizontalSpacer"></div>
                    <img src="assets/img/train/nav_icon_help_28.png">
                    <div class="navHorizontalSpacer"></div>
                    """),_display_(/*53.22*/i18n/*53.26*/.getMessage("train.nav.help")),format.raw/*53.55*/("""
                """),format.raw/*54.17*/("""</div>
                <div class="navBottom">
                    """),_display_(/*56.22*/i18n/*56.26*/.getMessage("train.nav.language")),format.raw/*56.59*/(""":
                    <select class="languageSelect" id="navLangSelect" onchange="changeLanguage('"""),_display_(/*57.98*/i18n/*57.102*/.getMessage("train.nav.langChangeErrorMsg")),format.raw/*57.145*/("""')">
                        <option value="en">English</option>
                        <option value="ja">日本語</option>
                        <option value="zh">中文</option>
                        <option value="kr">한글</option>
                    </select>
                    <div class="navElementSpacer"></div>
                    <a href=""""),_display_(/*64.31*/i18n/*64.35*/.getMessage("train.nav.websitelink")),format.raw/*64.71*/("""" class="textlink">"""),_display_(/*64.91*/i18n/*64.95*/.getMessage("train.nav.websitelinktext")),format.raw/*64.135*/("""</a>
                    <div class="navElementSpacer"></div>
                </div>
            </div>


            <div class="contentDiv" id="mainContentDiv">
                Content Div<br>
                Language: """),_display_(/*72.28*/i18n/*72.32*/.getDefaultLanguage),format.raw/*72.51*/("""
            """),format.raw/*73.13*/("""</div>
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
                  DATE: Sat Oct 15 18:44:54 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/Training.scala.html
                  HASH: 3ab1f263e3b575db871a1c956d81872ef50277c8
                  MATRIX: 588->1|721->39|749->41|955->221|967->225|1017->255|1385->595|1414->596|1456->610|1504->631|1517->635|1571->668|1652->722|1681->723|1714->729|2270->1258|2283->1262|2338->1295|2585->1515|2598->1519|2652->1552|2698->1570|2875->1720|2888->1724|2943->1757|3191->1978|3204->1982|3255->2012|3301->2030|3480->2182|3493->2186|3548->2219|3797->2441|3810->2445|3860->2474|3906->2492|4081->2640|4094->2644|4149->2677|4396->2897|4409->2901|4459->2930|4505->2948|4602->3018|4615->3022|4669->3055|4796->3155|4810->3159|4875->3202|5257->3557|5270->3561|5327->3597|5374->3617|5387->3621|5449->3661|5706->3891|5719->3895|5759->3914|5801->3928
                  LINES: 20->1|25->1|26->2|29->5|29->5|29->5|37->13|37->13|38->14|38->14|38->14|38->14|40->16|40->16|41->17|52->28|52->28|52->28|56->32|56->32|56->32|57->33|59->35|59->35|59->35|63->39|63->39|63->39|64->40|66->42|66->42|66->42|70->46|70->46|70->46|71->47|73->49|73->49|73->49|77->53|77->53|77->53|78->54|80->56|80->56|80->56|81->57|81->57|81->57|88->64|88->64|88->64|88->64|88->64|88->64|96->72|96->72|96->72|97->73
                  -- GENERATED --
              */
          