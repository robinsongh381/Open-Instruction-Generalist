#@title unnatural_instructions
"""
Copyright 2023, LAION contributors, inclduing Ontocord, LLC
and the other authors of OIG
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

import os
import json
from tqdm import tqdm
from datasets import load_dataset

from src.dataset import OIGBase


class UnnaturalInstructions(OIGBase):
    def __init__(self, output_path, hf_dataset_name="mrm8488/unnatural-instructions-full"):
        super().__init__()
        self.output_path = output_path
        self.hf_dataset_name = hf_dataset_name
    
    def __call__(self):
        instruction_with_output = []        
        dataset = load_dataset(self.hf_dataset_name)
                
        for i in tqdm(range(len(dataset['train']))):
            data = dataset['train'][i]
            for key in ['reformulations', 'instances']:
                d = data[key]
                
                if not d:
                    continue
                    
                if len(d) > 0:
                    instruction_with_output.extend([(ex['instruction_with_input'], ex['output']) for ex in d])
        
        output = []            
        instruction_with_output = list(set(instruction_with_output))
        
        for a, b in instruction_with_output:
            a = a.strip()
            a = a.replace("<sep>", "\n").replace("?", "?").replace("?", "?")
            b = b.strip()
            b = b.replace("<sep>", "\n").replace("?", "?").replace("?", "?")
            if b.count("?") == 1:
                if b[-1]  not in "?":
                    continue
            instance = {'text': "User: "+ a+"\nAssistant: "+ b, 'metadata': {'source': 'unatural_instructions'}}
            output.append(instance)

        # save output
        print('Length of output: ', len(output))
        self.save_output(output, self.output_path)