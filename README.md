# SynthPDF  
----
Generate Synthetic pdf files using custmizable content while recording elements coordinates for OCR objection detection, elements including table, stamp, paragraph, list, title, footnote, etc. All fully customizable with config file. Outputs pdf, img, and json for necessary information.



#![example1]

## Features

+ Highly costomizable yaml config file
+ Parallel running on multi-processors
+ Page elements: title, paragraph, table, space [customizable]
+ Various outputs
  - pdf
  - jpg (optional: with element bounding boxes)
  - json
+ Table:
  - nrows
  - ncols
  - alignment
  - space before/after
  - header content: Chinese / English / Decimals
  - table content: Chinese / English / Decimals / Specials
  - fonts / colors
  - with / without gridlines
  - many more ...
+ Paragraph
  - sentences
  - fonts / colors / words
  - alignment
  - linespace
+ Title:
  - Fonts / colors / words 
  - Number of lines
+ Spacer:
  - Spacer height / width

## Requirements

+ poppler-utils [linux]
+ reportlab==3.5.28 [python]
+ pdf2image==1.9.0 [python]
+ tqdm==4.35.0 [python]
+ matplotlib==3.1.1 [python]
+ numpuy==1.17.1 [python]
+ jsonschema==3.0.2 [python]
+ pyyaml==5.3.1 [python]
+ opencv==3.4.2 [python]


## Fonts

Below fonts are non-standard such that extracted bounding box is not accurate:
+ langting-gbk

## How to use
1. Clone the repo to local
2. Install required packages
via pip
```bash
pip install -r requirements.txt
```
or conda [recommended]
```bash
conda install --file requirements.txt -c conda-forge
```
3. Update config file (config.yaml) to custom the pdf layout

## TODO

+ Footnotes / pager header
+ horizontal lines
+ table stamps
+ rotate table
+ table line info to output json

