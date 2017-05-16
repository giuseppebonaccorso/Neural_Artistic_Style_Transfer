'''
Neural artistic styler

Based on: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, "A Neural Algorithm of Artistic Style", arXiv:1508.06576
Examples: https://www.bonaccorso.eu
See also: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Giuseppe Bonaccorso (https://www.bonaccorso.eu)
'''

from neural_styler import NeuralStyler

if __name__ == '__main__':
    print('Neural artistic styler')

    neural_styler = NeuralStyler(picture_image_filepath='img\\GB.jpg',
                                 style_image_filepath='img\\Magritte.jpg',
                                 destination_folder='\\destination_folder',

                                 # If you have a local copy of Keras VGG16/19 weights
                                 # weights_filepath='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

                                 alpha_picture=0.4,
                                 alpha_style=0.6,
                                 verbose=True,
                                 picture_layer='block4_conv1',
                                 style_layers=('block1_conv1',
                                               'block2_conv1',
                                               'block3_conv1',
                                               'block4_conv1',
                                               'block5_conv1'))

    # Create styled image
    neural_styler.fit(canvas='picture', optimization_method='L-BFGS-B')
    # or
    # neural_styler.fit(canvas='picture', optimization_method='CG')

    # Try also
    #
    # neural_styler.fit(canvas='random_from_style', optimization_method='L-BFGS-B')
    # and
    # neural_styler.fit(canvas='style')
    #
    # with different optimization algorithms (CG, etc.)

