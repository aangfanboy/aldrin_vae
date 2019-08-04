# **data_loader.py**
### Marshall
* load_image: `loads image, no label`
* read_all_data: `gets all the paths and labels, saves it as npy`

***Summary:*** Create tensorflow dataset object with map

#  **VAEmodel.py**
### Barney
* encoder_model: `create encoder model`
* decoder_model: `create decoder model`
* encode: `gets bottleneck and split into two(mean, logvar)`
* decode: `gets reparameterized input and put it into decoder`
* reparameterize: `gets mean and logvar and reparameterize them`
* generate_sample: `creates random input for decoder and generate image by through decoder`
* log_normal_pdf: `sub loss function`
* compute_loss: `do everything and calculate loss`
* save: `save encoder and decoder`
* train_step: `combine those things, and optimize model by through self.optimizer`

***Summary:*** Create model structure, loss, optimizer and simple train step


#  **main_trainer.py**
### Robin:
* save_images_to_tensorboard: `gets regenereted and real images, generate some from random and save to tensorboard`
* train_model: `train model by through barney and marshall, use 'save_images_to_tensorboard' to test it`

***Summary:*** Train model
