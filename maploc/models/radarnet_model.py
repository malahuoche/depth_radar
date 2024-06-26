import torch, torchvision
import numpy as np
from .networks import RadarNetV1Encoder,MultiScaleDecoder
from .base import BaseModel

class RadarNetModel(BaseModel):
    '''
    Image radar fusion to determine correspondence of radar to image

    Arg(s):
        input_channels_image : int
            number of channels in the image
        input_channels_depth : int
            number of channels in depth map
        input_patch_size_image : int
            patch of image to consider for radar point
        encoder_type : str
            encoder type
        n_filters_encoder_image : list[int]
            list of filters for each layer in image encoder
        n_neurons_encoder_image : list[int]
            list of neurons for each layer in depth encoder
        decoder_type : str
            decoder type
        n_filters_decoder : list[int]
            list of filters for each layer in decoder
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : str
            activation function for network
        device : torch.device
            device for running model
    '''
    default_conf = {
            "input_channels_image": 3,
            "input_channels_depth": 3,
            "input_patch_size_image": (512, 64),
            "encoder_type": "radarnetv1",
            "n_filters_encoder_image": [32, 64, 128, 128, 128],
            "n_neurons_encoder_depth": [32, 64, 128, 128, 128],
            "decoder_type": "multiscale",
            "n_filters_decoder": [256, 128, 64, 32, 16],
            "weight_initializer": "kaiming_uniform",
            "activation_func": "leaky_relu",
            "device": str(torch.device('cuda')),
        }



    def _init(self, conf):

        self.input_patch_size_image = self.conf.input_patch_size_image
        
        self.device = conf.device

        input_channels_image = conf.get("input_channels_image")
        input_channels_depth = conf.get("input_channels_depth")
        n_filters_encoder_image = conf.get("n_filters_encoder_image")
        n_neurons_encoder_depth = conf.get("n_neurons_encoder_depth")
        encoder_type = conf.get("encoder_type", 'radarnetv1')
        decoder_type = conf.get("decoder_type", 'multiscale')
        n_filters_decoder = conf.get("n_filters_decoder")
        weight_initializer = conf.get("weight_initializer", 'kaiming_uniform')
        activation_func = conf.get("activation_func", 'leaky_relu')

        # height, width = input_patch_size_image
        # latent_height = np.ceil(height / 32.0).astype(int)
        # latent_width = np.ceil(width / 32.0).astype(int)

        height, width = conf.input_patch_size_image
        latent_height = int((height // 32.0))
        latent_width = int((width // 32.0))

        latent_size_depth = latent_height * latent_width * n_neurons_encoder_depth[-1]

        # Build encoder
        if 'radarnetv1' in encoder_type:
            self.encoder = RadarNetV1Encoder(
                input_channels_image=input_channels_image,
                input_channels_depth=input_channels_depth,
                input_patch_size_image=self.conf.input_patch_size_image,
                n_filters_encoder_image=n_filters_encoder_image,
                n_neurons_encoder_depth=n_neurons_encoder_depth,
                latent_size_depth=latent_size_depth,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm='batch_norm' in encoder_type)
        else:
            raise ValueError('Encoder type {} not supported.'.format(encoder_type))

        # Calculate number of channels for latent and skip connections combining image + depth
        n_skips = n_filters_encoder_image[:-1]
        n_skips = n_skips[::-1] + [0]

        latent_channels = n_filters_encoder_image[-1] + n_neurons_encoder_depth[-1]

        # Build decoder
        if 'multiscale' in decoder_type:
            self.decoder = MultiScaleDecoder(
                input_channels=latent_channels,
                output_channels=1,
                n_resolution=1,
                n_filters=n_filters_decoder,
                n_skips=n_skips,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                output_func='linear',
                use_batch_norm='batch_norm' in decoder_type,
                deconv_type='up')
        else:
            raise ValueError('Decoder type {} not supported.'.format(decoder_type))

        # Move to device
        self.to(self.device)

    def _forward(self, data):
        '''
        Forwards the inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            point : torch.Tensor[float32]
                N x 3 input point
            return_logits : bool
                if set, then return logits otherwise sigmoid
        Returns:
            torch.Tensor[float32] : N x 1 x H x W logits (correspondence map)
        '''
        image = data['N_image']
        point = data['radar_points']
        bounding_boxes = data['bounding_boxes_list']
        # print("image 维度:", image.shape)
        # print("point 维度:", point.shape)
        # print("bounding_boxes 维度:", bounding_boxes.shape)
        return_logits = False
        latent, skips = self.encoder(image, point, bounding_boxes)

        logits = self.decoder(x=latent, skips=skips, shape=self.input_patch_size_image)[-1]
        return torch.sigmoid(logits)
        # if return_logits:
        #     return logits
        # else:
        #     return torch.sigmoid(logits)

    def compute_loss(self,
                     logits,
                     ground_truth,
                     validity_map,
                     w_positive_class=1.0):
        '''
        Computes loss function

        Arg(s):
            logits : torch.Tensor[float32]
                N x 1 x H x W logits
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations to compute loss
            w_positive_class : float
                weight of positive class
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        device = logits.device

        # Define loss function
        w_positive_class = torch.tensor(w_positive_class, device=device)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logits,
            target=ground_truth,
            reduction='none',
            pos_weight=w_positive_class)

        # Compute binary cross entropy
        loss = validity_map * loss
        loss = torch.sum(loss) / torch.sum(validity_map)

        loss_info = {
            'loss' : loss
        }

        return loss, loss_info


