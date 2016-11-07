
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
                        """),_display_(/*43.26*/i18n/*43.30*/.getMessage("train.session.label")),format.raw/*43.64*/("""
                        """),format.raw/*44.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*49.26*/i18n/*49.30*/.getMessage("train.session.worker.label")),format.raw/*49.71*/("""
                        """),format.raw/*50.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
                            <option>(Worker ID)</option>
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
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*66.137*/i18n/*66.141*/.getMessage("train.nav.overview")),format.raw/*66.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*67.104*/i18n/*67.108*/.getMessage("train.nav.model")),format.raw/*67.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*68.109*/i18n/*68.113*/.getMessage("train.nav.system")),format.raw/*68.144*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet">"""),_display_(/*69.102*/i18n/*69.106*/.getMessage("train.nav.userguide")),format.raw/*69.140*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*71.146*/i18n/*71.150*/.getMessage("train.nav.language")),format.raw/*71.183*/("""</span></a>
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
                                <h2><b>"""),_display_(/*107.41*/i18n/*107.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*107.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*111.55*/i18n/*111.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*111.110*/("""
                                    """),format.raw/*112.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*112.70*/i18n/*112.74*/.getMessage("train.overview.charts.iteration")),format.raw/*112.120*/("""
                                    """),format.raw/*113.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*121.41*/i18n/*121.45*/.getMessage("train.overview.perftable.title")),format.raw/*121.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*126.46*/i18n/*126.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*126.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*130.46*/i18n/*130.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*130.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*134.46*/i18n/*134.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*134.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*138.46*/i18n/*138.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*138.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*142.46*/i18n/*142.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*142.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*146.46*/i18n/*146.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*146.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*150.46*/i18n/*150.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*150.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*154.46*/i18n/*154.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*154.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*158.46*/i18n/*158.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*158.104*/("""</td>
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
                                <h2><b>"""),_display_(/*172.41*/i18n/*172.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*172.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*176.55*/i18n/*176.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*176.116*/("""
                                    """),format.raw/*177.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*178.47*/i18n/*178.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*178.108*/("""
                                    """),format.raw/*179.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*180.43*/i18n/*180.47*/.getMessage("train.overview.charts.iteration")),format.raw/*180.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*188.41*/i18n/*188.45*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*188.91*/(""": log<sub>10</sub></b></h2>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                    <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*190.156*/i18n/*190.160*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*190.216*/("""</a></li>
                                    <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*191.137*/i18n/*191.141*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*191.195*/("""</a></li>
                                    <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*192.133*/i18n/*192.137*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*192.189*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*197.55*/i18n/*197.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*197.110*/("""
                                    """),format.raw/*198.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*199.47*/i18n/*199.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*199.102*/("""
                                    """),format.raw/*200.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*201.43*/i18n/*201.47*/.getMessage("train.overview.charts.iteration")),format.raw/*201.93*/(""":</b> <span id="xStdev">
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
                $(document).ready(function () """),format.raw/*246.47*/("""{"""),format.raw/*246.48*/("""
                    """),format.raw/*247.21*/("""renderOverviewPage(true);
                """),format.raw/*248.17*/("""}"""),format.raw/*248.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*253.41*/("""{"""),format.raw/*253.42*/("""
                    """),format.raw/*254.21*/("""renderOverviewPage(false);
                """),format.raw/*255.17*/("""}"""),format.raw/*255.18*/(""", 2000);
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
                  DATE: Mon Nov 07 16:53:45 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 4ecefe6164f98da76ded4358992fed29880b1868
                  MATRIX: 604->1|737->39|765->41|888->138|900->142|950->172|2726->1921|2739->1925|2790->1955|2938->2076|2951->2080|3006->2114|3060->2140|3376->2429|3389->2433|3451->2474|3505->2500|4259->3226|4273->3230|4328->3263|4477->3384|4491->3388|4543->3418|4697->3544|4711->3548|4764->3579|4911->3698|4925->3702|4981->3736|5206->3933|5220->3937|5275->3970|7753->6420|7767->6424|7835->6470|8123->6730|8137->6734|8211->6785|8278->6823|8339->6856|8353->6860|8422->6906|8489->6944|8907->7334|8921->7338|8988->7383|9305->7672|9319->7676|9392->7726|9633->7939|9647->7943|9717->7991|9956->8202|9970->8206|10040->8254|10279->8465|10293->8469|10364->8518|10605->8731|10619->8735|10694->8787|10938->9003|10952->9007|11025->9057|11267->9271|11281->9275|11361->9332|11610->9553|11624->9557|11700->9610|11945->9827|11959->9831|12036->9885|12633->10454|12647->10458|12721->10510|13029->10790|13043->10794|13123->10851|13190->10889|13311->10982|13325->10986|13405->11043|13472->11081|13579->11160|13593->11164|13661->11210|14091->11612|14105->11616|14173->11662|14513->11973|14528->11977|14607->12033|14783->12180|14798->12184|14875->12238|15047->12381|15062->12385|15137->12437|15460->12732|15474->12736|15548->12787|15615->12825|15736->12918|15750->12922|15824->12973|15891->13011|15998->13090|16012->13094|16080->13140|18681->15712|18711->15713|18762->15735|18834->15778|18864->15779|19033->15919|19063->15920|19114->15942|19187->15986|19217->15987
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|67->43|67->43|67->43|68->44|73->49|73->49|73->49|74->50|90->66|90->66|90->66|91->67|91->67|91->67|92->68|92->68|92->68|93->69|93->69|93->69|95->71|95->71|95->71|131->107|131->107|131->107|135->111|135->111|135->111|136->112|136->112|136->112|136->112|137->113|145->121|145->121|145->121|150->126|150->126|150->126|154->130|154->130|154->130|158->134|158->134|158->134|162->138|162->138|162->138|166->142|166->142|166->142|170->146|170->146|170->146|174->150|174->150|174->150|178->154|178->154|178->154|182->158|182->158|182->158|196->172|196->172|196->172|200->176|200->176|200->176|201->177|202->178|202->178|202->178|203->179|204->180|204->180|204->180|212->188|212->188|212->188|214->190|214->190|214->190|215->191|215->191|215->191|216->192|216->192|216->192|221->197|221->197|221->197|222->198|223->199|223->199|223->199|224->200|225->201|225->201|225->201|270->246|270->246|271->247|272->248|272->248|277->253|277->253|278->254|279->255|279->255
                  -- GENERATED --
              */
          