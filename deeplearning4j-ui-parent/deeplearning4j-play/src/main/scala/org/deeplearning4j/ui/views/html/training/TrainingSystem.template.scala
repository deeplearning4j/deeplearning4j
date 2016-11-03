
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingSystem_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingSystem extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

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
                    <a class="brand" href="index.html"><span>"""),_display_(/*41.63*/i18n/*41.67*/.getMessage("train.pagetitle")),format.raw/*41.97*/("""</span></a>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*54.112*/i18n/*54.116*/.getMessage("train.nav.overview")),format.raw/*54.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*55.105*/i18n/*55.109*/.getMessage("train.nav.model")),format.raw/*55.139*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*56.138*/i18n/*56.142*/.getMessage("train.nav.system")),format.raw/*56.173*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet"> """),_display_(/*57.103*/i18n/*57.107*/.getMessage("train.nav.userguide")),format.raw/*57.141*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    Language</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
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

                    <!-- Start Content -->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span12">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*90.41*/i18n/*90.45*/.getMessage("train.system.title")),format.raw/*90.78*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <ul class="nav tab-menu nav-tabs" id="systemTab"></ul>

                                    <!--Start System Tab -->
                                <div id="myTabContent" class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- JVM Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span6">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*104.61*/i18n/*104.65*/.getMessage("train.system.chart.jvmTitle")),format.raw/*104.107*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="jvmmemorychart" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*108.75*/i18n/*108.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*108.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*109.58*/i18n/*109.62*/.getMessage("train.overview.charts.iteration")),format.raw/*109.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>
                                                <!-- Off Heap Memory Utlization Chart -->
                                            <div class="box span6">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*115.61*/i18n/*115.65*/.getMessage("train.system.chart.offHeapTitle")),format.raw/*115.111*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="offheapmemorychart" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*119.75*/i18n/*119.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*119.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*120.58*/i18n/*120.62*/.getMessage("train.overview.charts.iteration")),format.raw/*120.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*132.61*/i18n/*132.65*/.getMessage("train.system.hwTable.title")),format.raw/*132.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*138.70*/i18n/*138.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*138.120*/("""</th>
                                                                <th>"""),_display_(/*139.70*/i18n/*139.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*139.116*/("""</th>
                                                                <th>"""),_display_(/*140.70*/i18n/*140.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*140.124*/("""</th>
                                                                <th>"""),_display_(/*141.70*/i18n/*141.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*141.120*/("""</th>
                                                                <th>"""),_display_(/*142.70*/i18n/*142.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*142.118*/("""</th>
                                                                <th>"""),_display_(/*143.70*/i18n/*143.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*143.124*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="currentBytesJVM">Loading...</td>
                                                                <td id="maxBytesJVM">Loading...</td>
                                                                <td id="currentBytesOffHeap">Loading...</td>
                                                                <td id="maxBytesOffHeap">Loading...</td>
                                                                <td id="jvmAvailableProcessors">Loading...</td>
                                                                <td id="nComputeDevices">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                        <div class="row-fluid">

                                                <!-- Software Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*167.61*/i18n/*167.65*/.getMessage("train.system.swTable.title")),format.raw/*167.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*173.70*/i18n/*173.74*/.getMessage("train.system.swTable.hostname")),format.raw/*173.118*/("""</th>
                                                                <th>"""),_display_(/*174.70*/i18n/*174.74*/.getMessage("train.system.swTable.os")),format.raw/*174.112*/("""</th>
                                                                <th>"""),_display_(/*175.70*/i18n/*175.74*/.getMessage("train.system.swTable.osArch")),format.raw/*175.116*/("""</th>
                                                                <th>"""),_display_(/*176.70*/i18n/*176.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*176.117*/("""</th>
                                                                <th>"""),_display_(/*177.70*/i18n/*177.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*177.120*/("""</th>
                                                                <th>"""),_display_(/*178.70*/i18n/*178.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*178.121*/("""</th>
                                                                <th>"""),_display_(/*179.70*/i18n/*179.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*179.122*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="hostName">Loading...</td>
                                                                <td id="OS">Loading...</td>
                                                                <td id="OSArchitecture">Loading...</td>
                                                                <td id="jvmName">Loading...</td>
                                                                <td id="jvmVersion">Loading...</td>
                                                                <td id="nd4jBackend">Loading...</td>
                                                                <td id="nd4jDataType">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- GPU Chart -->
                                        <div class="row-fluid">
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>GPU Information (if isDevice == true)</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                                <th>?</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                                <td id="gpuPlaceholder">Loading...</td>
                                                            </tr>
                                                        </tbody>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                        <!-- End System Tab -->
                                </div>
                            </div>
                        </div>
                            <!-- End System Tab -->
                    </div><!-- End Row Fluid-->
                </div><!-- End Content -->
            </div><!-- End Container-->
        </div><!-- End Row Fluid-->

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
        <script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*279.47*/("""{"""),format.raw/*279.48*/("""
                    """),format.raw/*280.21*/("""renderSystemPage();
                    renderTabs();
                    selectMachine();
                """),format.raw/*283.17*/("""}"""),format.raw/*283.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*288.41*/("""{"""),format.raw/*288.42*/("""
                    """),format.raw/*289.21*/("""renderSystemPage();
                """),format.raw/*290.17*/("""}"""),format.raw/*290.18*/(""", 2000);
        </script>
            <!--End JavaScript-->

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
object TrainingSystem extends TrainingSystem_Scope0.TrainingSystem
              /*
                  -- GENERATED --
                  DATE: Thu Nov 03 20:03:32 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: 782db45fa7f259bedd71fd8171814d44b88e23c8
                  MATRIX: 600->1|733->39|761->41|884->138|896->142|946->172|2731->1930|2744->1934|2795->1964|3359->2500|3373->2504|3428->2537|3578->2659|3592->2663|3644->2693|3827->2848|3841->2852|3894->2883|4042->3003|4056->3007|4112->3041|6577->5479|6590->5483|6644->5516|7425->6269|7439->6273|7504->6315|7872->6655|7886->6659|7954->6704|8073->6795|8087->6799|8156->6845|8619->7280|8633->7284|8702->7330|9074->7674|9088->7678|9156->7723|9276->7815|9290->7819|9359->7865|9992->8470|10006->8474|10070->8515|10529->8946|10543->8950|10612->8996|10716->9072|10730->9076|10795->9118|10899->9194|10913->9198|10986->9248|11090->9324|11104->9328|11173->9374|11277->9450|11291->9454|11358->9498|11462->9574|11476->9578|11549->9628|13196->11247|13210->11251|13274->11292|13733->11723|13747->11727|13814->11771|13918->11847|13932->11851|13993->11889|14097->11965|14111->11969|14176->12011|14280->12087|14294->12091|14360->12134|14464->12210|14478->12214|14547->12260|14651->12336|14665->12340|14735->12387|14839->12463|14853->12467|14924->12515|21562->19124|21592->19125|21643->19147|21782->19257|21812->19258|21980->19397|22010->19398|22061->19420|22127->19457|22157->19458
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|78->54|78->54|78->54|79->55|79->55|79->55|80->56|80->56|80->56|81->57|81->57|81->57|114->90|114->90|114->90|128->104|128->104|128->104|132->108|132->108|132->108|133->109|133->109|133->109|139->115|139->115|139->115|143->119|143->119|143->119|144->120|144->120|144->120|156->132|156->132|156->132|162->138|162->138|162->138|163->139|163->139|163->139|164->140|164->140|164->140|165->141|165->141|165->141|166->142|166->142|166->142|167->143|167->143|167->143|191->167|191->167|191->167|197->173|197->173|197->173|198->174|198->174|198->174|199->175|199->175|199->175|200->176|200->176|200->176|201->177|201->177|201->177|202->178|202->178|202->178|203->179|203->179|203->179|303->279|303->279|304->280|307->283|307->283|312->288|312->288|313->289|314->290|314->290
                  -- GENERATED --
              */
          