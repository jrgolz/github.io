# Deep Learning Determination

Hello and welcome to this blog. Before the coronavirus pandemic had us sheltering place, I decided to take a deep learnings course at USF (virtual  of course). I have no computer science background, so I knew it would be tough for me. I have lots of interests, however, and wanted to add deep learnings to that list. My interests range from sports to sewing to parenting. I like to design physical things and create them on a 3D printer. I like being outside. My degree is in economics, my professions is in communications and I work for a VC firm. I'm all over the map. 

Before the course, I crammed on Python, which is compulsory know-how. 

My first project was a tree classifier model, which is suppose to discern between a Torrey pine and a blue spruce. Both are evergreens and I was interested to see if a computer vision model could understand all ospects of a tree - the shape, trunk, needles, cones, etc. It has been frustrating. I tried to use images from Bing. "Search_images_bing" is part of the fastai library, but my model said it was undefined. Fastai requires constant upgrading, but it never addressed the issue. The forums couldn't help either. I had similar issues with Google image search.

Finally iNaturalist worked. I downloaded hundreds of image urls for each tree and was able to convert to to jpegs. At this point I tried to merge what I had done with the code from the lesson. My next hurdle was at the parent_label, which I couldn't figure out. There are a set of functions related to downloading data and untaring data and identifying a path of the data, but I found that confusing. For example "path" doesn't show you the full path, and I couldn't figure out how to see it. That simple thing set me back. My workaround was to put all images in the same folder and just use Caps for Torrey Pine and lowercase for blue spruce. That way my model could know what was what using "isupper." This was how I defined my dataset. 

Next I use fastai's dataloader class to create batches of images to send to the GPU to 





You can include images:

![Image of fast.ai logo](images/logo.png)

## This is a title

And you can include links, like this [link to fast.ai](https://www.fast.ai). Posts will appear after this file. 
