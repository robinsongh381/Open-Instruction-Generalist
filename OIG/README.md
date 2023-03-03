## Script to Procee Dataset (WIP)

### Done Tasks
- [x] codeparrot_jupyter_summary
- [x] kojma_cot
- [x] unnatural_instructions
- [x] cuad
- [x] essays
- [x] mathqa
- [x] unified_skg
- [x] image_prompting_instructions # dataset_path
- [x] flanv2_cot
- [x] xp3_code

### Tasks with Issue
- poetry
  - Get error for `get_top_authors` function where poem does not have `author` key [code](https://github.com/LAION-AI/Open-Instruction-Generalist/blob/main/OIG/src/poetry.py#L123)
- soda_dialog
  - Get error for [this line](https://github.com/LAION-AI/Open-Instruction-Generalist/blob/main/OIG/src/soda_dialog.py#L51) because `dialog` is a list rather than being str 
  
### ToDo Tasks
- [] abstract_infill
- [] labeled_safety
- [] xp3


### Note
- Change the file name [merged_code_xp3.py](https://github.com/LAION-AI/Open-Instruction-Generalist/blob/main/OIG/src/merged_code_xp3.py) to xp3_code.py


### Process
```
python create_dataset.py --task_name essays --output_path processed/essays.json

```