### Basic facts:
1. **23** columns and **757.976** rows
2. **271** unique sample IDs in the dataset.
3. **444** unique Gene Names and **19004** unique Variant Ids
4. There are **4** types of patogenic variants: *['Wariant o nieznanej patogennosci', 'Wariant potencjalnie lagodny',
       'Wariant lagodny', 'Wariant potencjalnie patogenny',
       'Wariant patogenny']*
5. There are 3 classes to predict: **'braki zębowe' (1), 'rozszczepy' (2), 'kontrole' (0)**.

## Silent mutations 

- Silent mutations have low impact on patogenicity therefore will be removed.

- Original dateset shape is: 757976 rows
- Silent mutations shape is: 705564 rows,93.0% of all dataset.
- Non silent mutations shape is: 52412 rows, 7.0% of all dataset

**Number of patients 271**

#### Original dataset 

Number of patients per each patogenic variant
- Patogennosc
- Wariant lagodny                     271
- Wariant o nieznanej patogennosci    271
- Wariant patogenny                   258
- Wariant potencjalnie lagodny        271
- Wariant potencjalnie patogenny      270
- *Name: sample_ID, dtype: int64*

#### Only silent mutations 
Number of patients per each patogenic variant for only silent mutations
- Patogennosc
- Wariant lagodny                     271
- Wariant o nieznanej patogennosci    271
- Wariant patogenny                    34
- Wariant potencjalnie lagodny        271
- Wariant potencjalnie patogenny       34
- *Name: sample_ID, dtype: int64*

1. Conclusions: There is 34 patients with patogenic variant which are excluded due to silent mutations filter. 
2. Question: should we include them into the model?

# Variant Mild vs Maliscious 

- Mild variants shape has: 50498 rows
- Malicious variants shape has: 1914 rows

mild variants overview / numnber of patients
- target        Patogennosc                     
- **braki zębowe**
- Wariant lagodny                      66
- Wariant o nieznanej patogennosci     66
- Wariant potencjalnie lagodny         66
- **kontrole**    
- Wariant lagodny                     127
- Wariant o nieznanej patogennosci    127
- Wariant potencjalnie lagodny        127
- **rozszczepy**
- Wariant lagodny                      78
- Wariant o nieznanej patogennosci     78
- Wariant potencjalnie lagodny         78