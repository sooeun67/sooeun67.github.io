---
layout: single
title:  "Publish Jupyter Notebook to Github Blog as Jekyll Post"
categories:
  - Github
tags:
  - python
author_profile: false
---

#### I've been looking for an easy way to upload jupyter notebook as a Jekyll post without doing much modifications/styling

## 1. Convert your notebook to Markdown File

```python
jupyter nbconvert --to markdown your_jupyter_notebook.ipynb
```

## 2. Move your Markdown file to `_posts` folder

You need to have your markdown file ending with `.md` located in `_posts` under jekyll directory 

## 3. Add Jekyll metadata with layout
Below is a sample metadata with layout

```markdown
---
layout: single
title:  "Publish Jupyter Notebook to Github Blog as Jekyll Post"
categories:
  - Github
tags:
  - python
author_profile: false
---
```

## Conversion DONE :)