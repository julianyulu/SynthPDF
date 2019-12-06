# SynthPDF  
----
Generate Synthetic pdf files using custmizable content while recording elements coordinates for OCR objection detection

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

+ poppler-utils
+ reportlab==3.5.28
+ pdf2image==1.9.0
+ tqdm==4.35.0
+ matplotlib==3.1.1
+ numpuy==1.17.1
+ jsonschema==3.0.2

## Fonts

Below fonts are non-standard such that extracted bounding box is not accurate:
+ langting-gbk

## TODO

+ Footnotes / pager header
+ horizontal lines
+ table stamps
+ rotate table
+ table line info to output json

