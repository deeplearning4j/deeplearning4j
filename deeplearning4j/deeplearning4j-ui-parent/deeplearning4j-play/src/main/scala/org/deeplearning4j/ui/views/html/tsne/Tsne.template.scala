
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
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>
        <link src="/assets/webjars/bootstrap/2.3.2/css/bootstrap.min.css" rel="stylesheet" ></link>

            <!-- Latest compiled and minified JavaScript -->
        """),format.raw/*30.69*/("""
        """),format.raw/*31.9*/("""<script src="/assets/webjars/bootstrap/2.3.2/js/bootstrap.min.js"></script>


            <!-- d3 -->
        <script src="/assets/webjars/d3js/3.3.5/d3.min.js" charset="utf-8"></script>


"""),format.raw/*38.72*/("""
        """),format.raw/*39.9*/("""<script src="/assets/webjars/jquery-file-upload/8.4.2/js/jquery.fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/webjars/bootstrap-notify/3.1.3-1/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*51.13*/("""{"""),format.raw/*51.14*/("""
            """),format.raw/*52.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*55.9*/("""}"""),format.raw/*55.10*/("""

        """),format.raw/*57.9*/(""".block """),format.raw/*57.16*/("""{"""),format.raw/*57.17*/("""
            """),format.raw/*58.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""

        """),format.raw/*65.9*/(""".hd-small """),format.raw/*65.19*/("""{"""),format.raw/*65.20*/("""
            """),format.raw/*66.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*69.9*/("""}"""),format.raw/*69.10*/("""

        """),format.raw/*71.9*/("""body """),format.raw/*71.14*/("""{"""),format.raw/*71.15*/("""
            """),format.raw/*72.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*76.9*/("""}"""),format.raw/*76.10*/("""

        """),format.raw/*78.9*/("""#wrap """),format.raw/*78.15*/("""{"""),format.raw/*78.16*/("""
            """),format.raw/*79.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*82.9*/("""}"""),format.raw/*82.10*/("""

        """),format.raw/*84.9*/("""#embed """),format.raw/*84.16*/("""{"""),format.raw/*84.17*/("""
            """),format.raw/*85.13*/("""margin-top: 10px;
        """),format.raw/*86.9*/("""}"""),format.raw/*86.10*/("""

        """),format.raw/*88.9*/("""h1 """),format.raw/*88.12*/("""{"""),format.raw/*88.13*/("""
            """),format.raw/*89.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*91.9*/("""}"""),format.raw/*91.10*/("""

        """),format.raw/*93.9*/(""".tt """),format.raw/*93.13*/("""{"""),format.raw/*93.14*/("""
            """),format.raw/*94.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*98.9*/("""}"""),format.raw/*98.10*/("""

        """),format.raw/*100.9*/(""".txth """),format.raw/*100.15*/("""{"""),format.raw/*100.16*/("""
            """),format.raw/*101.13*/("""color: #F55;
        """),format.raw/*102.9*/("""}"""),format.raw/*102.10*/("""

        """),format.raw/*104.9*/(""".cit """),format.raw/*104.14*/("""{"""),format.raw/*104.15*/("""
            """),format.raw/*105.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*108.9*/("""}"""),format.raw/*108.10*/("""
        """),format.raw/*109.9*/("""</style>

        <script>
        $(document).ready(function () """),format.raw/*112.39*/("""{"""),format.raw/*112.40*/("""
            """),format.raw/*113.13*/("""$('#filenamebutton').click(function () """),format.raw/*113.52*/("""{"""),format.raw/*113.53*/("""
                """),format.raw/*114.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*120.13*/("""}"""),format.raw/*120.14*/(""");

            $('#form').fileUpload("""),format.raw/*122.35*/("""{"""),format.raw/*122.36*/("""
                """),format.raw/*123.17*/("""success: function (data, textStatus, jqXHR) """),format.raw/*123.61*/("""{"""),format.raw/*123.62*/("""
                    """),format.raw/*124.21*/("""var fullPath = document.getElementById('form').value;
                    var filename = data['name'];
                    if (fullPath) """),format.raw/*126.35*/("""{"""),format.raw/*126.36*/("""
                        """),format.raw/*127.25*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                        var filename = fullPath.substring(startIndex);
                        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*129.90*/("""{"""),format.raw/*129.91*/("""
                            """),format.raw/*130.29*/("""filename = filename.substring(1);
                        """),format.raw/*131.25*/("""}"""),format.raw/*131.26*/("""
                    """),format.raw/*132.21*/("""}"""),format.raw/*132.22*/("""

                    """),format.raw/*134.21*/("""document.getElementById('form').reset();

                    updateFileName(filename);
                    drawTsne();
                """),format.raw/*138.17*/("""}"""),format.raw/*138.18*/(""", error: function (err) """),format.raw/*138.42*/("""{"""),format.raw/*138.43*/("""
                    """),format.raw/*139.21*/("""console.log(err);
                    drawTsne();
                """),format.raw/*141.17*/("""}"""),format.raw/*141.18*/("""
            """),format.raw/*142.13*/("""}"""),format.raw/*142.14*/(""");


            function updateFileName(name) """),format.raw/*145.43*/("""{"""),format.raw/*145.44*/("""
                """),format.raw/*146.17*/("""$.ajax("""),format.raw/*146.24*/("""{"""),format.raw/*146.25*/("""
                    """),format.raw/*147.21*/("""url: '/tsne/upload',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify("""),format.raw/*150.42*/("""{"""),format.raw/*150.43*/(""""url": name"""),format.raw/*150.54*/("""}"""),format.raw/*150.55*/("""),
                    cache: false,
                    success: function (data, textStatus, jqXHR) """),format.raw/*152.65*/("""{"""),format.raw/*152.66*/("""
                        """),format.raw/*153.25*/("""setSessionId("UploadedFile");
                        drawTsne();
                    """),format.raw/*155.21*/("""}"""),format.raw/*155.22*/(""",
                    error: function (jqXHR, textStatus, errorThrown) """),format.raw/*156.70*/("""{"""),format.raw/*156.71*/("""
                        """),format.raw/*157.25*/("""// Handle errors here
                        console.log('ERRORS: ' + textStatus);
                        drawTsne();
                    """),format.raw/*160.21*/("""}"""),format.raw/*160.22*/("""
                """),format.raw/*161.17*/("""}"""),format.raw/*161.18*/(""");
            """),format.raw/*162.13*/("""}"""),format.raw/*162.14*/("""


        """),format.raw/*165.9*/("""}"""),format.raw/*165.10*/(""") ;

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

            """),format.raw/*220.53*/("""
                """),format.raw/*221.96*/("""
                """),format.raw/*222.42*/("""
                    """),format.raw/*223.59*/("""
                    """),format.raw/*224.68*/("""
                """),format.raw/*225.27*/("""
            """),format.raw/*226.23*/("""
        """),format.raw/*227.9*/("""</div>
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
                  DATE: Tue May 07 20:19:17 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: 75f5f005f3f45eea9992b7ae608a21cd60047aca
                  MATRIX: 634->0|1937->1335|1974->1345|2198->1612|2235->1622|2694->2053|2723->2054|2765->2068|2887->2163|2916->2164|2955->2176|2990->2183|3019->2184|3061->2198|3248->2358|3277->2359|3316->2371|3354->2381|3383->2382|3425->2396|3547->2491|3576->2492|3615->2504|3648->2509|3677->2510|3719->2524|3877->2655|3906->2656|3945->2668|3979->2674|4008->2675|4050->2689|4165->2777|4194->2778|4233->2790|4268->2797|4297->2798|4339->2812|4393->2839|4422->2840|4461->2852|4492->2855|4521->2856|4563->2870|4653->2933|4682->2934|4721->2946|4753->2950|4782->2951|4824->2965|4986->3100|5015->3101|5055->3113|5090->3119|5120->3120|5163->3134|5213->3156|5243->3157|5283->3169|5317->3174|5347->3175|5390->3189|5512->3283|5542->3284|5580->3294|5677->3362|5707->3363|5750->3377|5818->3416|5848->3417|5895->3435|6181->3692|6211->3693|6280->3733|6310->3734|6357->3752|6430->3796|6460->3797|6511->3819|6679->3958|6709->3959|6764->3985|7060->4252|7090->4253|7149->4283|7237->4342|7267->4343|7318->4365|7348->4366|7401->4390|7570->4530|7600->4531|7653->4555|7683->4556|7734->4578|7831->4646|7861->4647|7904->4661|7934->4662|8013->4712|8043->4713|8090->4731|8126->4738|8156->4739|8207->4761|8373->4898|8403->4899|8443->4910|8473->4911|8605->5014|8635->5015|8690->5041|8807->5129|8837->5130|8938->5202|8968->5203|9023->5229|9195->5372|9225->5373|9272->5391|9302->5392|9347->5408|9377->5409|9419->5423|9449->5424|11554->7540|11601->7637|11648->7680|11699->7740|11750->7809|11797->7837|11840->7861|11878->7871
                  LINES: 25->1|54->30|55->31|62->38|63->39|75->51|75->51|76->52|79->55|79->55|81->57|81->57|81->57|82->58|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|95->71|95->71|96->72|100->76|100->76|102->78|102->78|102->78|103->79|106->82|106->82|108->84|108->84|108->84|109->85|110->86|110->86|112->88|112->88|112->88|113->89|115->91|115->91|117->93|117->93|117->93|118->94|122->98|122->98|124->100|124->100|124->100|125->101|126->102|126->102|128->104|128->104|128->104|129->105|132->108|132->108|133->109|136->112|136->112|137->113|137->113|137->113|138->114|144->120|144->120|146->122|146->122|147->123|147->123|147->123|148->124|150->126|150->126|151->127|153->129|153->129|154->130|155->131|155->131|156->132|156->132|158->134|162->138|162->138|162->138|162->138|163->139|165->141|165->141|166->142|166->142|169->145|169->145|170->146|170->146|170->146|171->147|174->150|174->150|174->150|174->150|176->152|176->152|177->153|179->155|179->155|180->156|180->156|181->157|184->160|184->160|185->161|185->161|186->162|186->162|189->165|189->165|244->220|245->221|246->222|247->223|248->224|249->225|250->226|251->227
                  -- GENERATED --
              */
          