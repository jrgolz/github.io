# Deep Learning Determination

Hello and welcome to this blog. Before the coronavirus pandemic had us sheltering place, I decided to take a deep learnings course at USF. I have no computer science background, so I knew it would be tough for me. I have lots of interests, however, and wanted to add deep learnings to that list. My interests range from sports to sewing to parenting. I like to design physical things and create them on a 3D printer. I like being outside. My degree is in economics, my professions is in communications and I work for a VC firm. I'm all over the map. 

Before the course, I crammed on Python, which is compulsory know-how for deep learning. 

My first project was a computer vision model, which is suppose to discern between a Torrey pine and a blue spruce. Both are evergreen trees and I was interested to see if a model could understand all ospects of a tree - the shape, trunk, needles, cones, etc. It can as we'll see, but it was a frustrating first project. I first tried to get images from Bing. "Search_images_bing" is part of the fastai library, but my model repeatedly said it was undefined. Fastai requires constant upgrading, but it never addressed the issue. 

#### Sidebar: To avoid frustration from undefined code and strange errors, I importand and upgraded all these items at the start of each session:
```
from utils import *
from fastai2.vision.widgets import *
from fastai2.vision.all import *
import pandas as pd
import urllib.request
!pip install fastai2 --upgrade
!pip install fastcore --upgrade
```

The forums couldn't help with my image searches either. I had difficulties with Google too.

Finally iNaturalist worked. I downloaded hundreds of image urls for each tree type and was able to convert to to jpegs. I found this code to be able to do that: 

```
def url_to_jpg(i, url, path):
    filename = 'Torrey Pine-{}.jpg'.format(i)
    full_path = '{}{}'.format(path, filename)
    urllib.request.urlretrieve(url, full_path)
    
    print('{} saved.'.format(filename))
    return None
    
FILENAME = 'Torrey Pine urls.csv'
path = 'tree images/'

urls = pd.read_csv(FILENAME)

for i, url in enumerate(urls.values):
    url_to_jpg(i, url[0], path)
```

I repeated the code above to download blue spruces. In total I downloaded 1024 images. There is for sure a better way. 

At this point I tried to merge the download code I found with the code from the lesson. My next hurdle was at the parent_label, which I couldn't figure out. There are a set of functions related to downloading data and untaring data and identifying a path of the data, but I found that confusing. For example "path" doesn't show you the full path, and I couldn't figure out how to see it. That simple thing set me back. My workaround was to put all images in the same folder and just use Caps for Torrey Pine and lowercase for blue spruce. That way my model could know what was what using the "isupper method." Becasue of this, throughout my model, True = Torrey Pine. This was how I defined my dataset. 

Next I use fastai's dataloader class to create batches of images to send to the GPU to train the model.

```
def is_Torrey_Pine(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42, 
    label_func=is_Torrey_Pine, item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())
```

I tried a few different validation set sizes and item transforms and the ones above seemed to work best. I felt it was good enough for this project so I moved on to train the data, at first with resnet18, then resnet34 for its improved results: 

```
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)
```

Here was the output:

```
epoch	train_loss	valid_loss	error_rate	time
0	0.855714	0.284190	0.086538	00:04
epoch	train_loss	valid_loss	error_rate	time
0	0.271987	0.166655	0.057692	00:05
1	0.201745	0.059697	0.024038	00:05
2	0.150507	0.073935	0.024038	00:05
3	0.121402	0.077227	0.024038	00:05
```

It looks like there was a little overfitting, but I'm moving on. We can fix that later. 

My model was a little confused with blue spruces, thinking 4 of 96 were Torrey pines. Of 111 Torrey pines, only 1 was confused for a blue spurce. I've got no idea why this doesn't add up to 1024 images. 

For the exciting conclusion, I tested it with a blue spruce (a close up image of needles) with output below: 

```
learn_inf.predict('test images/blue spruce test.jpg')
('False', tensor(0), tensor([1.0000e+00, 2.9401e-08]))
```

And also a Torrey Pine (full profile of tree): 

```
learn_inf.predict('test images/Torrey Pine test.jpg')
('True', tensor(1), tensor([1.8352e-05, 9.9998e-01]))
```

It worked! If you recall from above, True = Torrey Pine. Fast forward to the GUI for the production site: 

```
def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
pred, pred_idx,probs = learn_inf.predict(img)
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_upload = widgets.FileUpload()

VBox([widgets.Label('Select your tree.'),
     btn_upload, btn_run, out_pl, lbl_pred])
```

Select your tree.

Prediction: True; Probability: 0.6025     

It took a lot of determination, but I created my first working computer vision model. The class is on week 6 and this is week 2 material. But I'm glad this is behind me. 

If I have time to build from here, I want to: 
1. Instead of hunting down a unique dataset, use an existing one to save a lot of headaches
2. Build a more sophisticated datablock
3. Try Regex so I could include blue spruces and several other tree types
4. Try a learning rate finder
5. Use multi-label clasification to create a more helpful model that somebody might really want to use

Here is what is holding me back: 
1. I don't know what path does. It feels like I should know by know but I don't. 
    a. Is it the path where to get urls online or is it the local path where things are on my drive? 
    b. Things worked when I repolicated ^path = untar_data^, but not when I used get_image_files from my local storage on Jupyter notebook. 
    c. Am I supposed to use ^path = untar_data...^ to pull image urls from inaturalist?
2. I don't know what the pre-requisites actions are for using regex. I tried to get it to work, but couldnt figure it out. 

