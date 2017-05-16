'''
Neural artistic styler

Based on: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, "A Neural Algorithm of Artistic Style", arXiv:1508.06576
Examples: https://www.bonaccorso.eu
See also: https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Giuseppe Bonaccorso (https://www.bonaccorso.eu)
'''

from __future__ import print_function

import numpy as np
import math
import keras.backend as K

from keras.applications import vgg16, vgg19
from keras.applications.imagenet_utils import preprocess_input

from scipy.optimize import minimize
from scipy.misc import imread, imsave, imresize

# Set random seed (for reproducibility)
np.random.seed(1000)


class NeuralStyler(object):
    def __init__(self, picture_image_filepath, style_image_filepath, destination_folder,
                 weights_filepath=None,
                 alpha_picture=1.0, alpha_style=0.00001,
                 save_every_n_steps=20,
                 verbose=False,
                 convnet='VGG16',
                 picture_layer='block5_conv1',
                 style_layers=('block1_conv1',
                               'block2_conv1',
                               'block3_conv1',
                               'block4_conv1',
                               'block5_conv1')):
        '''
        Artistic neural styler based on VGG16/19 convolutional network
        Based on: Leon A. Gatys, Alexander S. Ecker, Matthias Bethge,
                  "A Neural Algorithm of Artistic Style", arXiv:1508.06576
                  
        Parameters
        ----------
        :param picture_image_filepath: Content file path
        :param style_image_filepath: Style file path
        :param destination_folder: Result destination folder
        :param weights_filepath: Optional VGG19 weights filepath (HDF5)
        :param alpha_picture: Content loss function weight
        :param alpha_style: Style loss function weight
        :param save_every_n_steps: Save a picture every n optimization steps
        :param verbose: Print loss function values
        :param convnet: One of: VGG16 or VGG19
        :param picture_layer: Convnet layer used in the content loss function
        :param style_layers: Convnet layers used in the style loss function
        
        Usage examples
        ----------
        Picture and style over random:
            canvas='random_from_style'
            alpha_style=1.0
            alpha_picture=0.25
            picture_layer='block4_conv1' (both VGG16 and VGG19)
        Style over picture:
            canvas='picture'
            alpha_style=0.0025
            alpha_picture=1.0
            picture_layer='block4_conv1' (both VGG16 and VGG19)
        Picture over style:
            canvas='style'
            alpha_style=0.001
            alpha_picture=1.0
            picture_layer='block5_conv1' (both VGG16 and VGG19)
        '''

        if picture_image_filepath is None or style_image_filepath is None or destination_folder is None:
            raise ValueError('Picture, style image or destination filepath is/are missing')

        if convnet not in ('VGG16', 'VGG19'):
            raise ValueError('Convnet must be one of: VGG16 or VGG19')

        self.picture_image_filepath = picture_image_filepath
        self.style_image_filepath = style_image_filepath
        self.destination_folder = destination_folder

        self.alpha_picture = alpha_picture
        self.alpha_style = alpha_style
        self.save_every_n_steps = save_every_n_steps
        self.verbose = verbose

        self.layers = style_layers if picture_layer in style_layers else style_layers + (picture_layer,)

        self.iteration = 0
        self.step = 0
        self.styled_image = None

        # Create convnet
        print('Creating convolutional network')
        if convnet == 'VGG16':
            convnet = vgg16.VGG16(include_top=False, weights='imagenet' if weights_filepath is None else None)
        else:
            convnet = vgg19.VGG19(include_top=False, weights='imagenet' if weights_filepath is None else None)

        if weights_filepath is not None:
            print('Loading model weights from: %s' % weights_filepath)
            convnet.load_weights(filepath=weights_filepath)

        # Convnet output function
        self.get_convnet_output = K.function(inputs=[convnet.layers[0].input],
                                             outputs=[convnet.get_layer(t).output for t in self.layers])

        # Load picture image
        original_picture_image = imread(picture_image_filepath)

        self.image_shape = (original_picture_image.shape[0], original_picture_image.shape[1], 3)
        self.e_image_shape = (1,) + self.image_shape
        self.picture_image = self.pre_process_image(original_picture_image.reshape(self.e_image_shape).astype(K.floatx()))

        print('Loading picture: %s (%dx%d)' % (self.picture_image_filepath,
                                               self.picture_image.shape[2],
                                               self.picture_image.shape[1]))

        picture_tensor = K.variable(value=self.get_convnet_output([self.picture_image])[self.layers.index(picture_layer)])

        # Load style image
        original_style_image = imread(self.style_image_filepath)

        print('Loading style image: %s (%dx%d)' % (self.style_image_filepath,
                                                   original_style_image.shape[1],
                                                   original_style_image.shape[0]))

        # Check for style image size
        if (original_style_image.shape[0] != self.picture_image.shape[1]) or \
                (original_style_image.shape[1] != self.picture_image.shape[2]):
            # Resize image
            print('Resizing style image to match picture size: (%dx%d)' %
                  (self.picture_image.shape[2], self.picture_image.shape[1]))

            original_style_image = imresize(original_style_image,
                                            size=(self.picture_image.shape[1], self.picture_image.shape[2]),
                                            interp='lanczos')

        self.style_image = self.pre_process_image(original_style_image.reshape(self.e_image_shape).astype(K.floatx()))

        # Create style tensors
        style_outputs = self.get_convnet_output([self.style_image])
        style_tensors = [self.gramian(o) for o in style_outputs]

        # Compute loss function(s)
        print('Compiling loss and gradient functions')

        # Picture loss function
        picture_loss_function = 0.5 * K.sum(K.square(picture_tensor - convnet.get_layer(picture_layer).output))

        # Style loss function
        style_loss_function = 0.0
        style_loss_function_weight = 1.0 / float(len(style_layers))

        for i, style_layer in enumerate(style_layers):
            style_loss_function += \
                (style_loss_function_weight *
                (1.0 / (4.0 * (style_outputs[i].shape[1] ** 2.0) * (style_outputs[i].shape[3] ** 2.0))) *
                 K.sum(K.square(style_tensors[i] - self.gramian(convnet.get_layer(style_layer).output))))

        # Composite loss function
        composite_loss_function = (self.alpha_picture * picture_loss_function) + \
                                  (self.alpha_style * style_loss_function)

        loss_function_inputs = [convnet.get_layer(l).output for l in self.layers]
        loss_function_inputs.append(convnet.layers[0].input)

        self.loss_function = K.function(inputs=loss_function_inputs,
                                        outputs=[composite_loss_function])

        # Composite loss function gradient
        loss_gradient = K.gradients(loss=composite_loss_function, variables=[convnet.layers[0].input])

        self.loss_function_gradient = K.function(inputs=[convnet.layers[0].input],
                                                 outputs=loss_gradient)

    def fit(self, iterations=100, canvas='random', canvas_image_filepath=None, optimization_method='CG'):
        '''
        Create styled image
        :param iterations: Number of optimization iterations
        :param canvas: One of:
            'random': RGB random image
            'random_from_style': random image generated from style image pixels
            'random_from_picture': random image generated from picture pixels
            'style': Style image
            'picture': Picture
            'custom': Custom image specified by canvas_image paramater
        :param canvas_image_filepath: Canvas initial image file path
        :param optimization_method: Optimization method passed to SciPy optimize
            (See https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html
            for further information)
            Allowed options are:
                - Nelder-Mead
                - Powell
                - CG (Default)
                - BFGS
                - Newton-CG
                - L-BFGS-B
                - TNC
                - COBYLA
                - SLSQP
                - dogleg
                - trust-ncg
        '''

        if canvas not in ('random', 'random_from_style', 'random_from_picture', 'style', 'picture', 'custom'):
            raise ValueError('Canvas must be one of: random, random_from_style, '
                             'random_from_picture, style, picture, custom')

        # Generate random image
        if canvas == 'random':
            self.styled_image = self.pre_process_image(np.random.uniform(0, 256,
                                                                         size=self.e_image_shape).astype(K.floatx()))
        elif canvas == 'style':
            self.styled_image = self.style_image.copy()
        elif canvas == 'picture':
            self.styled_image = self.picture_image.copy()
        elif canvas == 'custom':
            self.styled_image = self.pre_process_image(imread(canvas_image_filepath).
                                                       reshape(self.e_image_shape).astype(K.floatx()))
        else:
            self.styled_image = np.ndarray(shape=self.e_image_shape)

            for x in range(self.picture_image.shape[2]):
                for y in range(self.picture_image.shape[1]):
                    x_p = np.random.randint(0, self.picture_image.shape[2] - 1)
                    y_p = np.random.randint(0, self.picture_image.shape[1] - 1)
                    self.styled_image[0, y, x, :] = \
                        self.style_image[0, y_p, x_p, :] if canvas == 'random_from_style' \
                        else self.picture_image[0, y_p, x_p, :]

        bounds = None

        # Set bounds if the optimization method supports them
        if optimization_method in ('L-BFGS-B', 'TNC', 'SLSQP'):
            bounds = np.ndarray(shape=(self.styled_image.flatten().shape[0], 2))
            bounds[:, 0] = -128.0
            bounds[:, 1] = 128.0

        print('Starting optimization with method: %r' % optimization_method)

        for _ in xrange(iterations):
            self.iteration += 1

            if self.verbose:
                print('Starting iteration: %d' % self.iteration)

            minimize(fun=self.loss, x0=self.styled_image.flatten(), jac=self.loss_gradient,
                     callback=self.callback, bounds=bounds, method=optimization_method)

            self.save_image(self.styled_image)

    def loss(self, image):
        outputs = self.get_convnet_output([image.reshape(self.e_image_shape).astype(K.floatx())])
        outputs.append(image.reshape(self.e_image_shape).astype(K.floatx()))

        v_loss = self.loss_function(outputs)[0]

        if self.verbose:
            print('\tLoss: %.2f' % v_loss)

        # Check whether loss has become NaN
        if math.isnan(v_loss):
            print('NaN Loss function value')

        return v_loss

    def loss_gradient(self, image):
        return np.array(self.loss_function_gradient([image.reshape(self.e_image_shape).astype(K.floatx())])).\
            astype('float64').flatten()

    def callback(self, image):
        self.step += 1
        self.styled_image = image.copy()

        if self.verbose:
            print('Optimization step: %d/%d' % (self.step, self.iteration))

        if self.step == 1 or self.step % self.save_every_n_steps == 0:
            self.save_image(image)

    def save_image(self, image):
        imsave(self.destination_folder + 'img_' + str(self.step) + '_' + str(self.iteration) + '.jpg',
               self.post_process_image(image.reshape(self.e_image_shape).copy()))

    @staticmethod
    def gramian(filters):
        c_filters = K.batch_flatten(K.permute_dimensions(K.squeeze(filters, axis=0), pattern=(2, 0, 1)))
        return K.dot(c_filters, K.transpose(c_filters))

    @staticmethod
    def pre_process_image(image):
        return preprocess_input(image)

    @staticmethod
    def post_process_image(image):
        image[:, :, :, 0] += 103.939
        image[:, :, :, 1] += 116.779
        image[:, :, :, 2] += 123.68
        return np.clip(image[:, :, :, ::-1], 0, 255).astype('uint8')[0]

