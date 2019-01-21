
package org.deeplearning4j.ui.views.html.tsne

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Tsne_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Tsne extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>T-SNE renders</title>
            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>


            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>


        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*54.13*/("""{"""),format.raw/*54.14*/("""
            """),format.raw/*55.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*58.9*/("""}"""),format.raw/*58.10*/("""

        """),format.raw/*60.9*/(""".block """),format.raw/*60.16*/("""{"""),format.raw/*60.17*/("""
            """),format.raw/*61.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*66.9*/("""}"""),format.raw/*66.10*/("""

        """),format.raw/*68.9*/(""".hd-small """),format.raw/*68.19*/("""{"""),format.raw/*68.20*/("""
            """),format.raw/*69.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*72.9*/("""}"""),format.raw/*72.10*/("""

        """),format.raw/*74.9*/("""body """),format.raw/*74.14*/("""{"""),format.raw/*74.15*/("""
            """),format.raw/*75.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*79.9*/("""}"""),format.raw/*79.10*/("""

        """),format.raw/*81.9*/("""#wrap """),format.raw/*81.15*/("""{"""),format.raw/*81.16*/("""
            """),format.raw/*82.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*85.9*/("""}"""),format.raw/*85.10*/("""

        """),format.raw/*87.9*/("""#embed """),format.raw/*87.16*/("""{"""),format.raw/*87.17*/("""
            """),format.raw/*88.13*/("""margin-top: 10px;
        """),format.raw/*89.9*/("""}"""),format.raw/*89.10*/("""

        """),format.raw/*91.9*/("""h1 """),format.raw/*91.12*/("""{"""),format.raw/*91.13*/("""
            """),format.raw/*92.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*94.9*/("""}"""),format.raw/*94.10*/("""

        """),format.raw/*96.9*/(""".tt """),format.raw/*96.13*/("""{"""),format.raw/*96.14*/("""
            """),format.raw/*97.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*101.9*/("""}"""),format.raw/*101.10*/("""

        """),format.raw/*103.9*/(""".txth """),format.raw/*103.15*/("""{"""),format.raw/*103.16*/("""
            """),format.raw/*104.13*/("""color: #F55;
        """),format.raw/*105.9*/("""}"""),format.raw/*105.10*/("""

        """),format.raw/*107.9*/(""".cit """),format.raw/*107.14*/("""{"""),format.raw/*107.15*/("""
            """),format.raw/*108.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*111.9*/("""}"""),format.raw/*111.10*/("""
        """),format.raw/*112.9*/("""</style>

        <script>
        $(document).ready(function () """),format.raw/*115.39*/("""{"""),format.raw/*115.40*/("""
            """),format.raw/*116.13*/("""$('#filenamebutton').click(function () """),format.raw/*116.52*/("""{"""),format.raw/*116.53*/("""
                """),format.raw/*117.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*123.13*/("""}"""),format.raw/*123.14*/(""");

            $('#form').fileUpload("""),format.raw/*125.35*/("""{"""),format.raw/*125.36*/("""
                """),format.raw/*126.17*/("""success: function (data, textStatus, jqXHR) """),format.raw/*126.61*/("""{"""),format.raw/*126.62*/("""
                    """),format.raw/*127.21*/("""var fullPath = document.getElementById('form').value;
                    var filename = data['name'];
                    if (fullPath) """),format.raw/*129.35*/("""{"""),format.raw/*129.36*/("""
                        """),format.raw/*130.25*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                        var filename = fullPath.substring(startIndex);
                        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*132.90*/("""{"""),format.raw/*132.91*/("""
                            """),format.raw/*133.29*/("""filename = filename.substring(1);
                        """),format.raw/*134.25*/("""}"""),format.raw/*134.26*/("""
                    """),format.raw/*135.21*/("""}"""),format.raw/*135.22*/("""

                    """),format.raw/*137.21*/("""document.getElementById('form').reset();

                    updateFileName(filename);
                    drawTsne();
                """),format.raw/*141.17*/("""}"""),format.raw/*141.18*/(""", error: function (err) """),format.raw/*141.42*/("""{"""),format.raw/*141.43*/("""
                    """),format.raw/*142.21*/("""console.log(err);
                    drawTsne();
                """),format.raw/*144.17*/("""}"""),format.raw/*144.18*/("""
            """),format.raw/*145.13*/("""}"""),format.raw/*145.14*/(""");


            function updateFileName(name) """),format.raw/*148.43*/("""{"""),format.raw/*148.44*/("""
                """),format.raw/*149.17*/("""$.ajax("""),format.raw/*149.24*/("""{"""),format.raw/*149.25*/("""
                    """),format.raw/*150.21*/("""url: '/tsne/upload',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify("""),format.raw/*153.42*/("""{"""),format.raw/*153.43*/(""""url": name"""),format.raw/*153.54*/("""}"""),format.raw/*153.55*/("""),
                    cache: false,
                    success: function (data, textStatus, jqXHR) """),format.raw/*155.65*/("""{"""),format.raw/*155.66*/("""
                        """),format.raw/*156.25*/("""setSessionId("UploadedFile");
                        drawTsne();
                    """),format.raw/*158.21*/("""}"""),format.raw/*158.22*/(""",
                    error: function (jqXHR, textStatus, errorThrown) """),format.raw/*159.70*/("""{"""),format.raw/*159.71*/("""
                        """),format.raw/*160.25*/("""// Handle errors here
                        console.log('ERRORS: ' + textStatus);
                        drawTsne();
                    """),format.raw/*163.21*/("""}"""),format.raw/*163.22*/("""
                """),format.raw/*164.17*/("""}"""),format.raw/*164.18*/(""");
            """),format.raw/*165.13*/("""}"""),format.raw/*165.14*/("""


        """),format.raw/*168.9*/("""}"""),format.raw/*168.10*/(""") ;

    </script>

    </head>

    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px;
                        text-align: right;" class="hd-small">&nbsp; Available sessions:
                        <select class="selectpicker" id="sessionSelect" onchange="selectNewSession()" style="color: #000000;
                            display: inline-block;
                            width: 256px;">
                            <option value="0" selected="selected">Pick a session to track</option>
                        </select>
                    </td>
                    <td style="width: 256px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>

        <br />
        <div style="text-align: center">
            <div id="embed" style="display: inline-block;
                width: 1024px;
                height: 700px;
                border: 1px solid #DEDEDE;"></div>
        </div>
        <br/>
        <br/>
        <div style="text-align: center;
            width: 100%;
            position: fixed;
            bottom: 0px;
            left: 0px;
            margin-bottom: 15px;">
            <div style="display: inline-block;
                margin-right: 48px;">
                <h5>Upload a file to UI server.</h5>
                <form encType="multipart/form-data" action="/tsne/upload" method="POST" id="form">
                    <div>

                        <input name="file" type="file" style="width: 300px;
                            display: inline-block;" />
                        <input type="submit" value="Upload file" style="display: inline-block;"/>

                    </div>
                </form>
            </div>

            """),format.raw/*223.53*/("""
                """),format.raw/*224.96*/("""
                """),format.raw/*225.42*/("""
                    """),format.raw/*226.59*/("""
                    """),format.raw/*227.68*/("""
                """),format.raw/*228.27*/("""
            """),format.raw/*229.23*/("""
        """),format.raw/*230.9*/("""</div>
    </body>

</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Tsne extends Tsne_Scope0.Tsne
              /*
                  -- GENERATED --
                  DATE: Mon Jan 21 11:34:13 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: 8c3701d9bbabb0a0f1fd606302a47f6c169bb85d
                  MATRIX: 634->0|2632->1970|2661->1971|2703->1985|2825->2080|2854->2081|2893->2093|2928->2100|2957->2101|2999->2115|3186->2275|3215->2276|3254->2288|3292->2298|3321->2299|3363->2313|3485->2408|3514->2409|3553->2421|3586->2426|3615->2427|3657->2441|3815->2572|3844->2573|3883->2585|3917->2591|3946->2592|3988->2606|4103->2694|4132->2695|4171->2707|4206->2714|4235->2715|4277->2729|4331->2756|4360->2757|4399->2769|4430->2772|4459->2773|4501->2787|4591->2850|4620->2851|4659->2863|4691->2867|4720->2868|4762->2882|4925->3017|4955->3018|4995->3030|5030->3036|5060->3037|5103->3051|5153->3073|5183->3074|5223->3086|5257->3091|5287->3092|5330->3106|5452->3200|5482->3201|5520->3211|5617->3279|5647->3280|5690->3294|5758->3333|5788->3334|5835->3352|6121->3609|6151->3610|6220->3650|6250->3651|6297->3669|6370->3713|6400->3714|6451->3736|6619->3875|6649->3876|6704->3902|7000->4169|7030->4170|7089->4200|7177->4259|7207->4260|7258->4282|7288->4283|7341->4307|7510->4447|7540->4448|7593->4472|7623->4473|7674->4495|7771->4563|7801->4564|7844->4578|7874->4579|7953->4629|7983->4630|8030->4648|8066->4655|8096->4656|8147->4678|8313->4815|8343->4816|8383->4827|8413->4828|8545->4931|8575->4932|8630->4958|8747->5046|8777->5047|8878->5119|8908->5120|8963->5146|9135->5289|9165->5290|9212->5308|9242->5309|9287->5325|9317->5326|9359->5340|9389->5341|11494->7457|11541->7554|11588->7597|11639->7657|11690->7726|11737->7754|11780->7778|11818->7788
                  LINES: 25->1|78->54|78->54|79->55|82->58|82->58|84->60|84->60|84->60|85->61|90->66|90->66|92->68|92->68|92->68|93->69|96->72|96->72|98->74|98->74|98->74|99->75|103->79|103->79|105->81|105->81|105->81|106->82|109->85|109->85|111->87|111->87|111->87|112->88|113->89|113->89|115->91|115->91|115->91|116->92|118->94|118->94|120->96|120->96|120->96|121->97|125->101|125->101|127->103|127->103|127->103|128->104|129->105|129->105|131->107|131->107|131->107|132->108|135->111|135->111|136->112|139->115|139->115|140->116|140->116|140->116|141->117|147->123|147->123|149->125|149->125|150->126|150->126|150->126|151->127|153->129|153->129|154->130|156->132|156->132|157->133|158->134|158->134|159->135|159->135|161->137|165->141|165->141|165->141|165->141|166->142|168->144|168->144|169->145|169->145|172->148|172->148|173->149|173->149|173->149|174->150|177->153|177->153|177->153|177->153|179->155|179->155|180->156|182->158|182->158|183->159|183->159|184->160|187->163|187->163|188->164|188->164|189->165|189->165|192->168|192->168|247->223|248->224|249->225|250->226|251->227|252->228|253->229|254->230
                  -- GENERATED --
              */
          