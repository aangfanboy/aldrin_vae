Generate cartoon faces with variational auto-encoders. Built with TensorFlow 2.0-beta1, trained 4 epochs with GTX1060 3GB and took 2 hours. Also in this software, you shall see a great use of TensorBoard. Here are results  generate with random bottlenecks;

![results](https://github.com/aangfanboy/aldrin_vae/blob/master/results.png)

You can download pre-trained models from the directory named `models`
You can download the dataset that i used from [here](https://google.github.io/cartoonset/)

Bottom, you can see files, classes in files, functions in classes and what those functions do. My class names are from TV Show named 'How I Met Your Mother', it is like this because i kinda like use the names of show characters on my code :)

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

[WARNING] This project still in developing faze which means that i will add more algorithms, data structures and stuff like that.

Please ask if you have questions :)
