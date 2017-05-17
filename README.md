# Neural artistic style tranfer
Based on: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, "A Neural Algorithm of Artistic Style", arXiv:1508.06576<br/>
See also: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

# Usage
There are three possibile canvas setup:
<ul>
<li><b>Picture</b>: The canvas is filled with the original picture</li>
<li><b>Style</b>: The canvas is filled with the style image (resized to match picture dimensions)</li>
<li><b>Random from style</b>: The canvas is filled with a random pattern generated starting from the style image</li>
</ul>
<p>
Some usage examples:
<br/><br/>
<b>Picture and style over random:</b><br/>
<i>canvas='random_from_style', alpha_style=1.0, alpha_picture=0.25, picture_layer='block4_conv1' (both VGG16 and VGG19)</i>
<br/>        
<b>Style over picture:</b><br/>
<i>canvas='picture', alpha_style=0.0025, alpha_picture=1.0, picture_layer='block4_conv1' (both VGG16 and VGG19)</i>
<br/>
<b>Picture over style:</b><br/>
<i>canvas='style', alpha_style=0.001, alpha_picture=1.0, picture_layer='block5_conv1' (both VGG16 and VGG19)</i>
</p>
<br/><br/>
# Code snippet
```python
neural_styler = NeuralStyler(picture_image_filepath='img\\GB.jpg',
                                 style_image_filepath='img\\Magritte.jpg',
                                 destination_folder='\\destination_folder',
                                 alpha_picture=0.4,
                                 alpha_style=0.6,
                                 verbose=True)

neural_styler.fit(canvas='picture', optimization_method='L-BFGS-B')
```

# Examples
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
</table>
