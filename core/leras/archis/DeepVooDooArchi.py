from core.leras import nn
tf = nn.tf

lowest_dense_res = 0

class DeepVooDooArchi(nn.ArchiBase):
    """
    resolution

    mod     None - default
            'quick'
    """
    def __init__(self, resolution, mod=None, opts=None):
        super().__init__()

        if opts is None:
            opts = ''

        if mod is None:
            class Downscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch, kernel_size=self.kernel_size, strides=2, padding='SAME')

                def forward(self, x):
                    x = self.conv1(x)
                    x = tf.nn.leaky_relu(x, 0.1)
                    return x

                def get_out_ch(self):
                    return self.out_ch

            class DownscaleBlock(nn.ModelBase):
                def on_build(self, in_ch, ch, n_downscales, kernel_size):
                    self.downs = []

                    last_ch = in_ch
                    for i in range(n_downscales):
                        cur_ch = ch*( min(2**i, 8)  )
                        self.downs.append ( Downscale(last_ch, cur_ch, kernel_size=kernel_size) )
                        last_ch = self.downs[-1].get_out_ch()

                def forward(self, inp):
                    x = inp
                    for down in self.downs:
                        x = down(x)
                    return x

            class Upscale(nn.ModelBase):
                def __init__(self, in_ch, out_ch, kernel_size=3, *kwargs ):
                    self.in_ch = in_ch
                    self.out_ch = out_ch
                    self.kernel_size = kernel_size
                    super().__init__(*kwargs)

                def on_build(self, *args, **kwargs ):
                    self.conv1 = nn.Conv2D( self.in_ch, self.out_ch*4, kernel_size=self.kernel_size, padding='SAME')
                    
                def forward(self, x):
                    x = self.conv1(x)
                    x = tf.nn.leaky_relu(x, 0.1)
                    x = nn.depth_to_space(x, 2)
                    return x

                def get_out_ch(self):
                    return self.out_ch

            class ResidualBlock(nn.ModelBase):
                def on_build(self, ch, kernel_size=3 ):
                    self.conv1 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')
                    self.conv2 = nn.Conv2D( ch, ch, kernel_size=kernel_size, padding='SAME')

                def forward(self, inp):
                    x = self.conv1(inp)
                    x = tf.nn.leaky_relu(x, 0.2)
                    x = self.conv2(x)
                    x = tf.nn.leaky_relu(inp + x, 0.2)
                    return x

            class Encoder(nn.ModelBase):
                def on_build(self, in_ch, e_ch, n_downscales=4):
                    global lowest_dense_res
                    lowest_dense_res = resolution // (2**(n_downscales + 1) if 'd' in opts else 2**n_downscales)

                    self.down1 = DownscaleBlock(in_ch, e_ch, n_downscales, kernel_size=5)

                def forward(self, inp):
                    return nn.flatten(self.down1(inp))

            
            class Inter(nn.ModelBase):
                def __init__(self, in_ch, ae_ch, ae_out_ch, num_layers, **kwargs):
                    self.in_ch, self.ae_ch, self.ae_out_ch, self.num_layers = in_ch, ae_ch, ae_out_ch, num_layers
                    super().__init__(**kwargs)


                def on_build(self):
                    global lowest_dense_res
                    in_ch, ae_ch, ae_out_ch = self.in_ch, self.ae_ch, self.ae_out_ch
                    if 'u' in opts:
                        self.dense_norm = nn.DenseNorm()

                    self.dense = []
                    last_ch = in_ch

                    for i in range(self.num_layers):
                        self.dense.append(nn.Dense( last_ch, ae_ch ))
                        last_ch = ae_ch
                    
                    self.dense.append(nn.Dense( ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch ))
                    self.upscale1 = Upscale(ae_out_ch, ae_out_ch)

                def forward(self, inp):
                    x = inp
                    if 'u' in opts:
                        x = self.dense_norm(x)

                    for d in self.dense:
                        x = d(x)
                        x = tf.nn.leaky_relu(x, 0.1)

                    x = nn.reshape_4D (x, lowest_dense_res, lowest_dense_res, self.ae_out_ch)
                    x = self.upscale1(x)

                    return x

                @staticmethod
                def get_code_res():
                    return lowest_dense_res

                def get_out_ch(self):
                    return self.ae_out_ch

            class Decoder(nn.ModelBase):

                def on_build(self, in_ch, d_ch, d_mask_ch, n_upscales=4 ):
                    self.ups = []
                    self.res = []
                    self.upms = [] 
                    last_ch = in_ch
                    last_ch_m = in_ch
                    kernel_size = 3

                    for i in reversed(range(1, n_upscales)):
                        cur_ch = d_ch *( min(2**i, 8)  )
                        m_cur_ch = d_mask_ch * ( min(2**i, 8) )
                        
                        self.ups.append( Upscale(last_ch, cur_ch, kernel_size=kernel_size) )
                        self.res.append( ResidualBlock(cur_ch, kernel_size=3) )
                        self.upms.append( Upscale(last_ch_m, m_cur_ch, kernel_size=kernel_size) )

                        last_ch = self.ups[-1].get_out_ch()
                        last_ch_m = self.upms[-1].get_out_ch()

                    self.out_conv  = nn.Conv2D( d_ch*2, 3, kernel_size=1, padding='SAME')
                    
                    if 'd' in opts:
                        self.out_convs = []
                        for i in reversed(range(1, n_upscales)):
                            self.out_convs.append(nn.Conv2D( d_ch*2, 3, kernel_size=kernel_size, padding='SAME'))
                        self.upscalem3 = Upscale(d_mask_ch*2, d_mask_ch*1, kernel_size=kernel_size)
                        self.out_convm = nn.Conv2D( d_mask_ch*1, 1, kernel_size=1, padding='SAME')
                    else:
                        self.out_convm = nn.Conv2D( d_mask_ch*2, 1, kernel_size=1, padding='SAME')

                def forward(self, inp):
                    z = inp
                    for up, res in zip(self.ups, self.res):
                        z = up(z)
                        z = res(z)
                    x = z

                    if 'd' in opts:

                        self.xs = []
                        self.xs.append(nn.upsample2d(tf.nn.sigmoid(self.out_conv(x))))
                        for out_conv in self.out_convs:
                            self.xs.append(nn.upsample2d(tf.nn.sigmoid(out_conv(x))))

                        if nn.data_format == "NHWC":
                            tile_cfg = ( 1, resolution // 2, resolution //2, 1)
                        else:
                            tile_cfg = ( 1, 1, resolution // 2, resolution //2 )

                        self.zs = []
                        for i in range(len(self.xs)):
                            z_cur =  tf.concat ( ( tf.concat ( (  tf.ones ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ),
                                                tf.concat ( ( tf.zeros ( (1,1,1,1) ), tf.zeros ( (1,1,1,1) ) ), axis=nn.conv2d_spatial_axes[1] ) ), axis=nn.conv2d_spatial_axes[0] )
                            self.zs.append(tf.tile ( z_cur, tile_cfg ))                       

                        for i, (xx, zz) in enumerate(zip(self.xs, self.zs)):
                            if i == 0:
                                x = xx * zz
                            else:
                                x += xx * zz
                    else:
                        x = tf.nn.sigmoid(self.out_conv(x))

                    zm = inp
                    for up in self.upms:
                        zm = up(zm)
                    m = zm

                    if 'd' in opts:
                        m = self.upscalem3(m)
                    m = tf.nn.sigmoid(self.out_convm(m))

                    return x, m
        
        self.Encoder = Encoder
        self.Inter = Inter
        self.Decoder = Decoder

nn.DeepVooDooArchi = DeepVooDooArchi