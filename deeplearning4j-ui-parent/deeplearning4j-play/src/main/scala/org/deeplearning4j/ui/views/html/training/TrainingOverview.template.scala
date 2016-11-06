
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingOverview_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingOverview extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->

    </head>

    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="#"><span>"""),_display_(/*41.54*/i18n/*41.58*/.getMessage("train.pagetitle")),format.raw/*41.88*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        Session:
                        <select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*60.137*/i18n/*60.141*/.getMessage("train.nav.overview")),format.raw/*60.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*61.104*/i18n/*61.108*/.getMessage("train.nav.model")),format.raw/*61.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*62.109*/i18n/*62.113*/.getMessage("train.nav.system")),format.raw/*62.144*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet">"""),_display_(/*63.102*/i18n/*63.106*/.getMessage("train.nav.userguide")),format.raw/*63.140*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*65.146*/i18n/*65.150*/.getMessage("train.nav.language")),format.raw/*65.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                    <!-- Start Score Chart-->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span8">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*101.41*/i18n/*101.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*101.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*105.55*/i18n/*105.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*105.110*/("""
                                    """),format.raw/*106.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*106.70*/i18n/*106.74*/.getMessage("train.overview.charts.iteration")),format.raw/*106.120*/("""
                                    """),format.raw/*107.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*115.41*/i18n/*115.45*/.getMessage("train.overview.perftable.title")),format.raw/*115.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*120.46*/i18n/*120.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*120.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*124.46*/i18n/*124.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*124.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*128.46*/i18n/*128.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*128.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*132.46*/i18n/*132.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*132.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*136.46*/i18n/*136.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*136.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*140.46*/i18n/*140.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*140.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*144.46*/i18n/*144.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*144.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*148.46*/i18n/*148.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*148.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*152.46*/i18n/*152.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*152.104*/("""</td>
                                        <td id="examplesPerSec">Loading...</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                            <!--End Model Table -->
                    </div>


                    <div class="row-fluid">
                            <!--Start Ratio Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*166.41*/i18n/*166.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*166.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*170.55*/i18n/*170.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*170.116*/("""
                                    """),format.raw/*171.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*172.47*/i18n/*172.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*172.108*/("""
                                    """),format.raw/*173.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*174.43*/i18n/*174.47*/.getMessage("train.overview.charts.iteration")),format.raw/*174.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b><h2><b>"""),_display_(/*182.48*/i18n/*182.52*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*182.98*/(""": log<sub>
                                    10</sub></b></h2></b></h2>
                                <div style="float: right">
                                    <p class="stackControls center">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*186.92*/i18n/*186.96*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*186.152*/("""" onclick="selectStdevChart('stdevActivations')">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*187.92*/i18n/*187.96*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*187.150*/("""" onclick="selectStdevChart('stdevGradients')">
                                        <input class="btn btn-small" type="button" value=""""),_display_(/*188.92*/i18n/*188.96*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*188.148*/("""" onclick="selectStdevChart('stdevUpdates')">
                                    </p>
                                </div>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*194.55*/i18n/*194.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*194.110*/("""
                                    """),format.raw/*195.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*196.47*/i18n/*196.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*196.102*/("""
                                    """),format.raw/*197.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*198.43*/i18n/*198.47*/.getMessage("train.overview.charts.iteration")),format.raw/*198.93*/(""":</b> <span id="xStdev">
                                        0</span></p>
                            </div>
                        </div>
                            <!-- End Variance Table -->
                    </div>
                </div>
            </div><!-- End Content Span10-->
        </div><!--End Row Fluid-->

        <!-- Start JavaScript-->
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
        <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
        <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
        <script src="/assets/js/jquery.ui.touch-punch.js"></script>
        <script src="/assets/js/modernizr.js"></script>
        <script src="/assets/js/bootstrap.min.js"></script>
        <script src="/assets/js/jquery.cookie.js"></script>
        <script src="/assets/js/fullcalendar.min.js"></script>
        <script src="/assets/js/jquery.dataTables.min.js"></script>
        <script src="/assets/js/excanvas.js"></script>
        <script src="/assets/js/jquery.flot.js"></script>
        <script src="/assets/js/jquery.flot.pie.js"></script>
        <script src="/assets/js/jquery.flot.stack.js"></script>
        <script src="/assets/js/jquery.flot.resize.min.js"></script>
        <script src="/assets/js/jquery.chosen.min.js"></script>
        <script src="/assets/js/jquery.uniform.min.js"></script>
        <script src="/assets/js/jquery.cleditor.min.js"></script>
        <script src="/assets/js/jquery.noty.js"></script>
        <script src="/assets/js/jquery.elfinder.min.js"></script>
        <script src="/assets/js/jquery.raty.min.js"></script>
        <script src="/assets/js/jquery.iphone.toggle.js"></script>
        <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
        <script src="/assets/js/jquery.gritter.min.js"></script>
        <script src="/assets/js/jquery.imagesloaded.js"></script>
        <script src="/assets/js/jquery.masonry.min.js"></script>
        <script src="/assets/js/jquery.knob.modified.js"></script>
        <script src="/assets/js/jquery.sparkline.min.js"></script>
        <script src="/assets/js/counter.js"></script>
        <script src="/assets/js/retina.js"></script>
        <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*243.47*/("""{"""),format.raw/*243.48*/("""
                    """),format.raw/*244.21*/("""renderOverviewPage();
                """),format.raw/*245.17*/("""}"""),format.raw/*245.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*250.41*/("""{"""),format.raw/*250.42*/("""
                    """),format.raw/*251.21*/("""renderOverviewPage();
                """),format.raw/*252.17*/("""}"""),format.raw/*252.18*/(""", 2000);
        </script>
            <!-- End JavaScript-->

    </body>
</html>
"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingOverview extends TrainingOverview_Scope0.TrainingOverview
              /*
                  -- GENERATED --
                  DATE: Sun Nov 06 14:54:16 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: f1668f3c3677b1216add7e7c4931e781094ffead
                  MATRIX: 604->1|737->39|765->41|888->138|900->142|950->172|2726->1921|2739->1925|2790->1955|3701->2838|3715->2842|3770->2875|3919->2996|3933->3000|3985->3030|4139->3156|4153->3160|4206->3191|4353->3310|4367->3314|4423->3348|4648->3545|4662->3549|4717->3582|7195->6032|7209->6036|7277->6082|7565->6342|7579->6346|7653->6397|7720->6435|7781->6468|7795->6472|7864->6518|7931->6556|8349->6946|8363->6950|8430->6995|8747->7284|8761->7288|8834->7338|9075->7551|9089->7555|9159->7603|9398->7814|9412->7818|9482->7866|9721->8077|9735->8081|9806->8130|10047->8343|10061->8347|10136->8399|10380->8615|10394->8619|10467->8669|10709->8883|10723->8887|10803->8944|11052->9165|11066->9169|11142->9222|11387->9439|11401->9443|11478->9497|12075->10066|12089->10070|12163->10122|12471->10402|12485->10406|12565->10463|12632->10501|12753->10594|12767->10598|12847->10655|12914->10693|13021->10772|13035->10776|13103->10822|13540->11231|13554->11235|13622->11281|13947->11578|13961->11582|14040->11638|14210->11780|14224->11784|14301->11838|14469->11978|14483->11982|14558->12034|14960->12408|14974->12412|15048->12463|15115->12501|15236->12594|15250->12598|15324->12649|15391->12687|15498->12766|15512->12770|15580->12816|18181->15388|18211->15389|18262->15411|18330->15450|18360->15451|18529->15591|18559->15592|18610->15614|18678->15653|18708->15654
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|84->60|84->60|84->60|85->61|85->61|85->61|86->62|86->62|86->62|87->63|87->63|87->63|89->65|89->65|89->65|125->101|125->101|125->101|129->105|129->105|129->105|130->106|130->106|130->106|130->106|131->107|139->115|139->115|139->115|144->120|144->120|144->120|148->124|148->124|148->124|152->128|152->128|152->128|156->132|156->132|156->132|160->136|160->136|160->136|164->140|164->140|164->140|168->144|168->144|168->144|172->148|172->148|172->148|176->152|176->152|176->152|190->166|190->166|190->166|194->170|194->170|194->170|195->171|196->172|196->172|196->172|197->173|198->174|198->174|198->174|206->182|206->182|206->182|210->186|210->186|210->186|211->187|211->187|211->187|212->188|212->188|212->188|218->194|218->194|218->194|219->195|220->196|220->196|220->196|221->197|222->198|222->198|222->198|267->243|267->243|268->244|269->245|269->245|274->250|274->250|275->251|276->252|276->252
                  -- GENERATED --
              */
          