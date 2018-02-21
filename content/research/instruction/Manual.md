+++
title = "A Manual on How To Write a Blog Post"
date = '2018-02-19'
tags = ["Class17/18"]
categories = ["instruction"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Class of Winter Term 2017 / 2018"
disqusShortname = "https-humbodt-wi-github-io-blog"
description = " A Manual on How To Write a Blog Post"
+++

The Website of the Institute of Information Systems is based on a framework called Hugo. Hugo is a static website generator, which allows us to easily present your content of your deep learning projects. In order to present results in the best way, we combine this tool with another tool called Gist. Within this documentation you will learn how to work with those tools.
<br>
The manual is composed of following sections:

##### 1. Getting Started: Installation of Hugo framework
##### 2. Getting Started: Github
##### 3. Hugo Basics
##### 4. How to create a blog post with Markdown
##### 5. Code Integration Tool: Gist

Besides this manual, we also created a [sample post](https://humboldt-wi.github.io/blog/research/instruction/00samplepost/) based on materials from group NN Fundamentals for your reference.

<br>
## 1) Getting Started: Installation of Hugo framework
<br>
Hugo offers a good manual on how to install all the necessary files depending on which operating system you use. Follow these links:
<br>
<br>
https://gohugo.io/getting-started/installing/

https://gohugo.io/getting-started/quick-start/
<br>
<br>

After a successful installation, you can download the complete content of the website via Github in step 2).
<br>
<br>

## 2) Getting Started: Github
<br>
All of the data that is used by hugo to build the website is stored in our Github repository. Use the following link:
<br>
<br>
https://github.com/Humboldt-WI/blog
<br>
<br>

You can upload your post following the similiar steps from the instruction by Johannes.


Please

* Fork the repository (create a copy on your account)
* Clone the repository (create a copy of the files on your computer)
* Save all your post and images in the corresponding folders
* Create a pull request
<br>

Clone the respository to your local device. This can easily be done through git. The short command for that is shown here:

```
git clone git@github.com:your_account/blog.git
```

If you are in need of a more detailed explanation of how to clone this repository, please follow this link:
<br>
<br>
https://git-scm.com/book/it/v2/Git-Basics-Getting-a-Git-Repository
<br>
<br>
At this point, you should now have all the files of the website on you local device. In 3) you will learn, how to use Hugo to build, change and develop your blog post.
<br>
<br>
## 3) Hugo Basics
<br>
Open your git bash (if it isn’t already open) and make sure you are in the correct folder. In this case the folder should be called ``blog`` and what you should see should look like the following folder structure.
<br>
<br>
```
blog
├── .git
├── archetypes
├── config.toml
├── content
├── data
├── layouts
├── static
└── themes
```
<br>
<br>
With the following command, you are building your Hugo website in draft mode and are able to view it in your browser, while it is still running on a local host
<br>
<br>
```
hugo server -D
```
<br>
<br>

The following code tells you that your web server is available at
<br>
<br>
http://localhost:1313/blog/
<br>
<br>
```
$ hugo server -D
Started building sites ...

Built site for language en:
0 draft content
0 future content
0 expired content
20 regular pages created
52 other pages created
0 non-page files copied
22 paginator pages created
14 tags created
1 categories created
3 month created
total in 38 ms
Watching for changes in /Users/Documents/Blog/{data,content,layouts,themes,static}
Serving pages from memory
Running in Fast Render Mode. For full rebuilds on change: hugo server --disableFastRender
Web Server is available at 127.0.0.1:1313 (bind address 127.0.0.1)
Press Ctrl+C to stop
```
<br>
<br>
Quit this running process with ``Ctrl + C``, if you are done or want to restart building the website. The later can be necessary after inserting folders or images.

Not every folder is relevant for you as an editor (Hugo of course needs all of them, so do not delete the other ones). If you are only interested in creating a blog post, then you only have to be concerned about ``content`` and ``static``. If you are curious for information about all of the folders follow this link. If not, just continue reading.
<br>
<br>
https://gohugo.io/getting-started/directory-structure/
<br>
<br>
 ``content`` as the name says is where the files are stored for each site of the overall website. Hugo works with markdown files, which you will learn in more detail in 4).

When working with Hugo it is essential to know where which information is stored. Within the content folder, you can see two folders and two markdown files. The folders represent a group of sites. The mardown files represent a single site of the website. From these files, Hugo is building the menu of the website.

File structure of the content folder:
<br>
<br>
```
content
├── member
├── news
├── research
├── contact.md
└── contributors.md
```
<br>
<br>

Website Look:
<br>
<br>
![](/blog/img/manual/website.png)
<br>
<br>
Going deeper, within the folder ``research`` you can find corresponding pages that present results of the chair of information systems. We distinguish here between work that has been conducted within the seminar and work outside of the seminar, i.e. as a results of dissertations. At the current time the later folder is empty and is only a proposal.
<br>
<br>
```
content
├── member
├── news
├── research
│   ├── _index.md
│   ├── seminar
│   └── instruction
├── contact.md
└── contributors.md
```
<br>
<br>
Within the seminar folder you can find markdown files already created for your respective topics.
<br>
<br>
```
research
├── _index.md
├── seminar
│   ├── 01NeuralNetworkFundamentals.md
│   ├── 02ConvolutionalNeuralNetworks.md
│   ├── 03ImageAnalysis.md
│   ├── 04TopicModels.md
│   ├── 05SentimentAnalysis.md
│   ├── 06FinancialTime Series.md
│   ├── 07ImageCaptioning.md
│   └── 08Recommendation.md
└── instruction
```
<br>
<br>
Website look:
<br>
<br>
![](/blog/img/manual/seminar_folder.png)
<br>
<br>
From an marketing point of view it is good to make use of the following features:


* Visually appealing images that fit well to the content of the post
* Catchy headlines that create curiosity about the subject
* Adding tags to refer to certain topics (just like hashtags)



<br>
## 4) How to create a blog post with Markdown
<br>
You can use any text editor to edit your Markdown file.
(I personally use Atom. Some packages, such as Markdown Writer and Markdown Preview, make it easier to write Markdown.)

Here you can find a simple guide of Markdown syntax:
[Mastering Markdown](https://guides.github.com/features/mastering-markdown/
)

As we mentioned in the last section, we have already created a draft for each group in the seminar folder.
Once opening your draft, you will see the header as following:
<br>
<br>
```
+++
title = "Sample Post"
date = '2017-12-14'
tags = [ "Deep Learning", "Neural Networks", "Class17/18",]
categories = ["instruction"]
banner = "img/seminar/sample/hu-logo.jpg"
author = "Class of Winter Term 2017 / 2018"
disqusShortname = "https-wisample-github-io-blog"
description = " A Sample Post"
+++
```
<br>
<br>
Although we have pre-defined some of the variables in the header, there are many variables you still need to modify.
<br>
<b>Varibles have to be modified:</b>

* ``date``
  The time you publish your post; should follow the format 'YYYY-MM-DD'
* ``description``
  One sentence description of your post
* ``banner``A image represents your topic the most; will show up in the content list page; details see below

<b>Varibles can be modified:</b>

* ``title``
  The title of your post.
* ``tags``
  We have predined tags for each topic. You can add some, if you feel neccessary.

<b>Varibles don't need to be modified:</b>

* ``categories``
* ``author``
* ``disqusShortname``

<br>
#### How to customize the banner and other images in your post:

You can see the variable ``banner`` displays as following in the header:
```
banner = "img/seminar/sample/hu-logo.jpg"
```
In order to customize the banner, you should change the path ``img/seminar/sample/hu-logo.jpg``

All images are saved under the ``static`` section.
We have created image folders for each group in the ``seminar`` folder under ``static``.

File structure of the static folder:
<br>
```
static
└── img
    ├── banner
    ├── carousel
    ├── manual
    ├── news
    ├── seminar
    │   ├── CNN
    │   ├── financial_time_series
    │   ├── image_analysis
    │   ├── image_captioning
    │   ├── nn_fundamentals
    │   ├── recommendation
    │   ├── sample
    │   ├── sentiment_analysis
    │   └── topic_models
    ├── sign
    ├── team
    └── teaching
```
<br>
All the pictures in your blog post should be saved in the corresponding folder.
<br>
Thus, the ``banner`` should be modified as
``
banner = "img/seminar/your_group_folder/your_image.jpg"
``
in the header.

<br>
<br>
#### Some highlights of Markdown Syntax
#### Image

In order to embed images in the markdown file, you should do the following:
<br>
```
![](/blog/img/seminar/your_group_folder/your_image.jpg)
```
Slightly different from the ``banner``, you also have to include ``/blog`` before the path ``/img``.
<br>
<br>
For example:
<br>
```
![example](/blog/img/seminar/sample/hu-logo.jpg)
```
<br>
<br>
![example](/blog/img/seminar/sample/hu-logo.jpg)

<br>
<br>
You can use HTML to control the size and alignment of the image
<br>
<br>
```
![hu logo]<img align="center" width="200" height="200"  src="/blog/img/seminar/sample/hu-logo.jpg">
```
<br>
<br>
<img align="center" width="200" height="200"  src="/blog/img/seminar/sample/hu-logo.jpg">
<br>
<br>
<br>
<br>
#### Blank Line

As I mentioned before you can use some HTML in Markdown. To add blank line within texts, you can simply add ``<br>`` in your file.

For example:
<br>
<br>
```
##### Inspect the Data
It seems like all the elements of our neural network structure are ready. Can we use this structure to tackle the problem?
Let's take a look at our data.
<br>
<br>
1,2,3,4,5
<br>
<br>
##### Fit draft of the NN to the Data
```
<br>
<br>

> ##### Inspect the Data
> It seems like all the elements of our neural network structure are ready. Can we use this structure to tackle the problem?
>Let's take a look at our data.
><br>
><br>
>1,2,3,4,5
><br>
><br>
>##### Fit draft of the NN to the Data


For more about Markdown Syntax, please follow the links:

 [Mastering Markdown](https://guides.github.com/features/mastering-markdown/
 )

[Markdown Syntax](https://daringfireball.net/projects/markdown/)

https://en.wikipedia.org/wiki/Markdown
<br>
<br>
## 5) Code Integration Tool: Gist

Exporting the Jupyter Notebook into Gist involves following steps:

1. Download the ipynb file
2. Open it with a text editor
3. Copy the content
4. Paste the content into Gist (remember to include the .ipynb extension in the filename)

You also can use <b>the Unofficial Jupyter Notebook Extensions</b> to create Gist automatically.
<br>
<br>
More details here:
[jupyter-contrib-nbextensions](http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/gist_it/readme.html)

The Gist can be embedded in the Markdown file as following:
<br>
```
<script src="https://gist.github.com/HsueanL/7356b55ef381ee05012c798be0c6ef2a.js"></script>
```
<br>
It looks like:
<br>
<br>
<script src="https://gist.github.com/HsueanL/7356b55ef381ee05012c798be0c6ef2a.js"></script>

To better understand how Markdown and Gist work, you can further check the [sample post](https://humboldt-wi.github.io/blog/research/instruction/00samplepost/).
The Markdown files for this post and the sample post are saved in the instrucion folder.
<br>
```
content
├── member
├── news
├── research
│   ├── _index.md
│   ├── seminar
│   └── instruction
│       ├──00SamplePost.md
│       └──Manual.md
├── contact.md
└── contributors.md
```
<br>
