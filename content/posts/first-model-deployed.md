---
title: "Deep Learning Journey : it took 5 years to finally train and deploy a model"
date: 2023-02-13T09:03:20-08:00
summary: "As so many people, I discovered the field of data science with the well known Machine Learning course
by Andrew Ng on Coursera. It was great. Years later, after following the Deep Learning Specialization and many other courses, I still was not able to deploy a model by myself."
draft: false
---

A note for the potential millions of readers : I feel a discomfort to write this post. In my head it sounds like: me me me, and me again. But as it is recommended by the fastai community, I will try the experience, to see. Sorry in advance if it sounds too much like this.

## A developer discovers the Machine Learning

Just to put some context, I'm Nicolas, a 37 yo French guy born in Nice, I have studied biology, ending up to work as a developer for few years now. Also I like very much to travel.  
Then I discoverd the field of AI and wanted to know more technically about it. It's where I discovered the Machine Learning course of Andrew Ng. I loved it.  


## Master skill in Courses Certifications, Zero Skill in doing something

And it's where started my infinite tutorial hole that already many articles describe. Yes I could write from scratch all the forward and back propagation. It felt nice for the ego: yeah, I know what is under the hood.  
Well, great Nicolas. Then I registered for my first Kaggle competition (other than the Titanic and price houses)
I couldn't do anything by myself. Load the data properly, clean the data, data augmentation, validation/test sets. I knew all this in theory, I knew how it was working and why to do it, but I could'nt effectively implement it.  
And I started to look at some Kaggle notebooks. The number of notebooks, the complexity of them, it was so overwhelming that I stopped and didn't come back for more than one year haha.

## An Architecture Classifier

Here is the result of the lesson 2. I will not share or explain here the entire code, it's pointless as it's already done in the notebooks and video of the course.  
It's just to share some context for the architecture context.

For the selection of the types of architecture, I just look over internet a little, but by no mean it means they are the best or anything.

```python
searches = ['baroche architecture', 'gothic architecture',
'byzantine architecture', 'postmodern architecture', 'greek and roman classical architecture',
'neoclassical architecture', 'victorian architecture','modern architecture']
path = Path('architecture')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    resize_images(path/o, max_size=400, dest=path/o)

```

![png](/first-model-deployed/batch-images.png)
    

```python
learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(5)
```




```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(6, 6))
```


<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>



![png](/first-model-deployed/confusion-matrix.png)

At the end the model is working not badly (the accuracy is not very high, 70%, but enough for me). And it's quite funny to put in Hugging Face Space some church, museum or any building pictures to check the architecture.  
I mean, the feeling is nice to have built something concrete.

The links:  
[Kaggle notebook](https://www.kaggle.com/code/nicolasca/blog-post-lesson-2-fastai)  
[Hugging Face Space](https://huggingface.co/spaces/nicolasca/architecture-classifier)

## The fastai experience

I shared what I did, even it's not that interesting by itself (I just copied what is done during the video).  
I could have done a dog/wolf classifier, a poo classifier (well..maybe...one day...let's keep that in mind), it's not important.  
What is important is that I did something as an end to end process. From 0. Yes I used a very high level lib (fastai) where I don't need to understand anything on what's done behind, yes I copied/pasted some code. Yep, it's true. **But** at the end I have created a notebook with jupyter on local env, also on Kaggle for the GPU traning, I have learned how to use gradio and Hugging Face. And most importantly I have something to show: an architecture classifier that is working not badly. And this is satisfaying.  

**It's the first time since I have started to learn Machine Learning that I was able to build anything in the real world**. That feel nice ! And it doesn't matter nobody is using it. I feel like a kid building some Lego. The result doesn't matter, the process of building by itself is enough !

Anyway I loved the Coursera ML and DL courses. Really it's high quality. But it is not enough, and the fastai approach of top-bottom is very interesting. Both approaches are complementary, we should do both, both theory and practice. What I'm saying is soooo obivous, but it took me some time to really do it (I think because of lazyness).

## Then after ?
No idea. Maybe I will finish the course, start to be quite skilled and why not work in the field. Maybe I will stop after the lesson 3. Maybe will I have a break and start again in 2 years. It's all possible.
It doesn't really matter, what's important is to enjoy it, out of any stress, any ambition, any results. Just to enjoy it for what it is, here building something.

I wish to all of you the best. Not only in your journey in deep learning, this is secondary, but in your life. 
May you be able to find what you really like in life. 

Nicolas, with :heart: