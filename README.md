# Neural artistic style tranfer
<img src="https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000"/><br/>

Based on: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, <i>"<a href="https://arxiv.org/abs/1508.06576" target="_blank">A Neural Algorithm of Artistic Style</a>"</i>, arXiv:1508.06576<br/>
See also: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py<br/>
See some examples on: https://www.bonaccorso.eu/2016/11/13/neural-artistic-style-transfer-experiments-with-keras/

## Usage
There are three possibile canvas setup:
<ul>
<li><b>Picture</b>: The canvas is filled with the original picture</li>
<li><b>Style</b>: The canvas is filled with the style image (resized to match picture dimensions)</li>
<li><b>Random from style</b>: The canvas is filled with a random pattern generated starting from the style image</li>
</ul>
<p>
Some usage examples (both with VGG16 and VGG19):
<br/><br/>
<b>Picture and style over random:</b><br/>
<i>canvas='random_from_style', alpha_style=1.0, alpha_picture=0.25, picture_layer='block4_conv1'</i>
<br/>        
<b>Style over picture:</b><br/>
<i>canvas='picture', alpha_style=0.0025, alpha_picture=1.0, picture_layer='block4_conv1'</i>
<br/>
<b>Picture over style:</b><br/>
<i>canvas='style', alpha_style=0.001, alpha_picture=1.0, picture_layer='block5_conv1'</i>
</p>

For a mix of style transfer and deepdream generation, see the examples below.

## Code snippets
```
neural_styler = NeuralStyler(picture_image_filepath='img\\GB.jpg',
                                 style_image_filepath='img\\Magritte.jpg',
                                 destination_folder='\\destination_folder',
                                 alpha_picture=0.4,
                                 alpha_style=0.6,
                                 verbose=True)

neural_styler.fit(canvas='picture', optimization_method='L-BFGS-B')
```

```
neural_styler = NeuralStyler(picture_image_filepath='img\\GB.jpg',
                                 style_image_filepath='img\\Magritte.jpg',
                                 destination_folder='\\destination_folder',
                                 alpha_picture=0.25,
                                 alpha_style=1.0,
                                 picture_layer='block4_conv1',
                                 style_layers=('block1_conv1',
                                               'block2_conv1',
                                               'block3_conv1',
                                               'block4_conv1',
                                               'block5_conv1'))
                                               
neural_styler.fit(canvas='random_from_style', optimization_method='CG')
```

## Examples
(With different settings and optimization algorithms)
<table width="100%" align="center">
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Cezanne.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Cezanne</b></p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Magritte.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Magritte</b></p>
</td>
</tr>
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Dal%C3%AC.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Dalì</b></p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Matisse.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Matisse</b></p>
</td>
</tr>
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Picasso.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Picasso</b></p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Rembrandt.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Rembrandt</b></p>
</td>
</tr>
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/De+Chirico.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>De Chirico</b></p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Mondrian.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Mondrian</b></p>
</td>
</tr>
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Van+Gogh.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Van Gogh</b></p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Schiele.jpg" align="center" height="600" width="338">
</p>
<p align="center"><b>Schiele</b></p>
</td>
</tr>
</table>

## Mixing style transfer and deep dreams
I'm still working on some experiments based on loss function which tries to maximize the L2 norm of the last convolutional block (layers 1 and 2). I've excluded those from the style_layers tuple and tuned the parameters to render a "dream" together with a styled image. You can try the following snippet:
```
# Dream loss function
dream_loss_function = -5.0*K.sum(K.square(convnet.get_layer('block5_conv1').output)) + \
                      -2.5*K.sum(K.square(convnet.get_layer('block5_conv2').output))

# Composite loss function
composite_loss_function = (self.alpha_picture * picture_loss_function) + \
                          (self.alpha_style * style_loss_function) + \
                          dream_loss_function
```
The composite loss function isnt't "free" to maximize the norm like in <a href="https://github.com/giuseppebonaccorso/keras_deepdream">Keras DeepDream</a>, because the MSE with the gramian terms forces the filters to get similar to the style, however, it's possible to obtain interesting results. The following pictures show the famous Tübingen styled with a Braque painting and forced to render "random" elements (they're similar to animal heads and eyes) like in a dream:

<table width="100%" align="center">
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Tubinga.jpg" align="center" width="400">
</p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Tubinga_dream.jpg" align="center" width="400">
</p>
</td>
</tr>
</table>

This example, instead, has been created using a VGG19 with a Cezanne painting and:
```
style_layers=('block1_conv1',
              'block2_conv1',
              'block3_conv1',
              'block4_conv1',
              'block5_conv1',
              'block5_conv2')
              
# Dream loss function
dream_loss_function = -10.0*K.sum(K.square(convnet.get_layer('block5_conv1').output)) + \
                      -5.0*K.sum(K.square(convnet.get_layer('block5_conv2').output))
```

(Original image by Manfred Brueckels - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=6937538)

<table width="100%" align="center">
<tr>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Park.jpg" align="center" width="400">
</p>
</td>
<td width="auto">
<p align="center">
<img src="https://s3-us-west-2.amazonaws.com/neural-style-transfer-demo/Park_Dream.jpg" align="center" width="400">
</p>
</td>
</tr>
</table>


## Requirements
<ul>
<li>Python 2.7-3.5</li>
<li>Keras</li>
<li>Theano/Tensorflow</li>
<li>SciPy</li>
</ul>
